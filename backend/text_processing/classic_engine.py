import math
import torch

from collections import namedtuple
from backend.text_processing import parsing, emphasis
from backend.text_processing.textual_inversion import EmbeddingDatabase
from backend import memory_management


PromptChunkFix = namedtuple('PromptChunkFix', ['offset', 'embedding'])
last_extra_generation_params = {}


class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []
        self.fixes = []


class CLIPEmbeddingForTextualInversion(torch.nn.Module):
    def __init__(self, wrapped, embeddings, textual_inversion_key='clip_l'):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings
        self.textual_inversion_key = textual_inversion_key
        self.weight = self.wrapped.weight

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None

        inputs_embeds = self.wrapped(input_ids)

        if batch_fixes is None or len(batch_fixes) == 0 or max([len(x) for x in batch_fixes]) == 0:
            return inputs_embeds

        vecs = []
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                emb = embedding.vec[self.textual_inversion_key] if isinstance(embedding.vec, dict) else embedding.vec
                emb = emb.to(inputs_embeds)
                emb_len = min(tensor.shape[0] - offset - 1, emb.shape[0])
                tensor = torch.cat([tensor[0:offset + 1], emb[0:emb_len], tensor[offset + 1 + emb_len:]]).to(dtype=inputs_embeds.dtype)

            vecs.append(tensor)

        return torch.stack(vecs)


class ClassicTextProcessingEngine:
    def __init__(
            self, text_encoder, tokenizer, chunk_length=75,
            embedding_dir=None, embedding_key='clip_l', embedding_expected_shape=768, emphasis_name="Original",
            text_projection=False, minimal_clip_skip=1, clip_skip=1, return_pooled=False, final_layer_norm=True
    ):
        super().__init__()

        self.embeddings = EmbeddingDatabase(tokenizer, embedding_expected_shape)

        if isinstance(embedding_dir, str):
            self.embeddings.add_embedding_dir(embedding_dir)
            self.embeddings.load_textual_inversion_embeddings()

        self.embedding_key = embedding_key

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        self.emphasis = emphasis.get_current_option(emphasis_name)()
        self.text_projection = text_projection
        self.minimal_clip_skip = minimal_clip_skip
        self.clip_skip = clip_skip
        self.return_pooled = return_pooled
        self.final_layer_norm = final_layer_norm

        self.chunk_length = chunk_length

        self.id_start = self.tokenizer.bos_token_id
        self.id_end = self.tokenizer.eos_token_id
        self.id_pad = self.tokenizer.pad_token_id

        self.text_encoder_model = getattr(text_encoder, "transformer", text_encoder)
        self.text_model = getattr(
            self.text_encoder_model,
            "text_model",
            self.text_encoder_model,
        )

        model_embeddings = self.text_model.embeddings
        model_embeddings.token_embedding = CLIPEmbeddingForTextualInversion(
            model_embeddings.token_embedding,
            self.embeddings,
            textual_inversion_key=embedding_key,
        )

        vocab = self.tokenizer.get_vocab()

        self.comma_token = vocab.get(',</w>', None)

        self.token_mults = {}

        tokens_with_parens = [(k, v) for k, v in vocab.items() if '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult

    def empty_chunk(self):
        chunk = PromptChunk()
        chunk.tokens = [self.id_start] + [self.id_end] * (self.chunk_length + 1)
        chunk.multipliers = [1.0] * (self.chunk_length + 2)
        return chunk

    def get_target_prompt_token_count(self, token_count):
        return math.ceil(max(token_count, 1) / self.chunk_length) * self.chunk_length

    def tokenize(self, texts):
        tokenized = self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

        return tokenized

    def encode_with_transformers(self, tokens):
        target_device = memory_management.text_encoder_device()

        # self.text_encoder.transformer.text_model.embeddings.position_ids = self.text_encoder.transformer.text_model.embeddings.position_ids.to(device=target_device)
        # self.text_encoder.transformer.text_model.embeddings.position_embedding = self.text_encoder.transformer.text_model.embeddings.position_embedding.to(dtype=torch.float32)
        # self.text_encoder.transformer.text_model.embeddings.token_embedding = self.text_encoder.transformer.text_model.embeddings.token_embedding.to(dtype=torch.float32)

        text_encoder_model = self.text_encoder_model
        text_model = self.text_model
        embeddings = text_model.embeddings

        embeddings.position_ids = embeddings.position_ids.to(device=target_device)
        embeddings.position_embedding = embeddings.position_embedding.to(dtype=torch.float32)
        embeddings.token_embedding = embeddings.token_embedding.to(dtype=torch.float32)

        tokens = tokens.to(target_device)
        outputs = text_encoder_model(tokens, output_hidden_states=True)

        layer_id = -max(self.clip_skip, self.minimal_clip_skip)
        z = outputs.hidden_states[layer_id]

        if self.final_layer_norm:
            z = text_model.final_layer_norm(z)

        if self.return_pooled:
            pooled_output = outputs.pooler_output

            if self.text_projection:
                pooled_output = text_encoder_model.text_projection(pooled_output)

            z.pooled = pooled_output
            
        return z

    def tokenize_line(self, line):
        parsed = parsing.parse_prompt_attention(line)
        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0
        last_comma = -1
        comma_padding_backtrack = 20

        def next_chunk(is_last=False):
            nonlocal chunk
            nonlocal token_count
            nonlocal last_comma

            # 避免连续 BREAK 或边界检查产生空 chunk。
            # 空文本仍需要保留一个合法的空 chunk。
            if not chunk.tokens and chunks and not is_last:
                last_comma = -1
                return False

            if len(chunk.tokens) > self.chunk_length:
                raise ValueError(
                    f"CLIP chunk is too long: {len(chunk.tokens)} > {self.chunk_length}"
                )

            original_length = len(chunk.tokens)
            token_count += original_length

            padding_count = self.chunk_length - original_length
            if padding_count:
                chunk.tokens.extend([self.id_end] * padding_count)
                chunk.multipliers.extend([1.0] * padding_count)

            # CLIP 输入长度固定为 chunk_length + BOS + EOS。
            chunk.tokens = [self.id_start] + chunk.tokens + [self.id_end]
            chunk.multipliers = [1.0] + chunk.multipliers + [1.0]

            if len(chunk.tokens) != self.chunk_length + 2:
                raise RuntimeError(
                    f"Invalid CLIP chunk length: {len(chunk.tokens)}"
                )

            chunks.append(chunk)
            chunk = PromptChunk()
            last_comma = -1
            return True

        def move_recent_tokens_to_next_chunk():
            nonlocal chunk
            nonlocal last_comma

            if last_comma < 0:
                return False

            if len(chunk.tokens) - last_comma > comma_padding_backtrack:
                return False

            moved_tokens = chunk.tokens[last_comma + 1:]
            moved_multipliers = chunk.multipliers[last_comma + 1:]

            chunk.tokens = chunk.tokens[:last_comma + 1]
            chunk.multipliers = chunk.multipliers[:last_comma + 1]

            next_chunk(is_last=False)
            chunk.tokens = moved_tokens
            chunk.multipliers = moved_multipliers
            return True

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == "BREAK" and weight == -1:
                if chunk.tokens:
                    next_chunk(is_last=False)
                continue

            position = 0

            while position < len(tokens):
                embedding, embedding_length = self.embeddings.find_embedding_at_position(
                    tokens,
                    position,
                )

                if embedding is None:
                    if len(chunk.tokens) >= self.chunk_length:
                       next_chunk(is_last=False)

                    token = tokens[position]
                    chunk.tokens.append(token)
                    chunk.multipliers.append(weight)
                    position += 1

                    if token == self.comma_token:
                        last_comma = len(chunk.tokens) - 1

                    if len(chunk.tokens) >= self.chunk_length:
                        if not move_recent_tokens_to_next_chunk():
                            next_chunk(is_last=False)

                    continue

                embedding_length = int(embedding_length)
                embedding_vectors = int(embedding.vectors)

                if len(chunk.tokens) + embedding_vectors > self.chunk_length:
                    next_chunk(is_last=False)

                chunk.fixes.append(
                    PromptChunkFix(
                        offset=len(chunk.tokens),
                        embedding=embedding,
                    )
                )
                chunk.tokens.extend([0] * embedding_vectors)
                chunk.multipliers.extend([weight] * embedding_vectors)
                position += embedding_length

                if len(chunk.tokens) >= self.chunk_length:
                    next_chunk(is_last=False)

        if chunk.tokens or not chunks:
            next_chunk(is_last=True)

        return chunks, token_count

    def process_texts(self, texts):
        token_count = 0

        cache = {}
        batch_chunks = []
        for line in texts:
            if line in cache:
                chunks = cache[line]
            else:
                chunks, current_token_count = self.tokenize_line(line)
                token_count = max(current_token_count, token_count)

                cache[line] = chunks

            batch_chunks.append(chunks)

        return batch_chunks, token_count

    def __call__(self, texts):
        batch_chunks, token_count = self.process_texts(texts)

        used_embeddings = {}
        chunk_count = max([len(x) for x in batch_chunks])

        zs = []
        for i in range(chunk_count):
            batch_chunk = [chunks[i] if i < len(chunks) else self.empty_chunk() for chunks in batch_chunks]

            tokens = [x.tokens for x in batch_chunk]
            multipliers = [x.multipliers for x in batch_chunk]
            self.embeddings.fixes = [x.fixes for x in batch_chunk]

            for fixes in self.embeddings.fixes:
                for _position, embedding in fixes:
                    used_embeddings[embedding.name] = embedding

            z = self.process_tokens(tokens, multipliers)
            zs.append(z)

        global last_extra_generation_params

        if used_embeddings:
            names = []

            for name, embedding in used_embeddings.items():
                print(f'[Textual Inversion] Used Embedding [{name}] in CLIP of [{self.embedding_key}]')
                names.append(name.replace(":", "").replace(",", ""))

            if "TI" in last_extra_generation_params:
                last_extra_generation_params["TI"] += ", " + ", ".join(names)
            else:
                last_extra_generation_params["TI"] = ", ".join(names)

        if any(x for x in texts if "(" in x or "[" in x) and self.emphasis.name != "Original":
            last_extra_generation_params["Emphasis"] = self.emphasis.name

        if self.return_pooled:
            return torch.hstack(zs), zs[0].pooled
        else:
            return torch.hstack(zs)

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        tokens = torch.asarray(remade_batch_tokens)

        if self.id_end != self.id_pad:
            for batch_pos in range(len(remade_batch_tokens)):
                index = remade_batch_tokens[batch_pos].index(self.id_end)
                tokens[batch_pos, index + 1:tokens.shape[1]] = self.id_pad

        z = self.encode_with_transformers(tokens)

        pooled = getattr(z, 'pooled', None)

        self.emphasis.tokens = remade_batch_tokens
        self.emphasis.multipliers = torch.asarray(batch_multipliers).to(z)
        self.emphasis.z = z
        self.emphasis.after_transformers()
        z = self.emphasis.z

        if pooled is not None:
            z.pooled = pooled

        return z
