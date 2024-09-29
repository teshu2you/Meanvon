import torch

from backend.huggingface_guess import model_list
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.vae import VAE
from backend.patcher.unet import UnetPatcher
from backend.text_processing.kolors_engine import KolorsTextProcessingEngine
from backend.args import dynamic_args
from backend import memory_management
from backend.nn.unet import Timestep

class Kolors(ForgeDiffusionEngine):
    matched_guesses = [model_list.Kolors]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(
            model_dict={
                'clip_l': huggingface_components['text_encoder']
            },
            tokenizer_dict={
                'clip_l': huggingface_components['tokenizer']
            }
        )

        vae = VAE(model=huggingface_components['vae'])

        unet = UnetPatcher.from_model(
            model=huggingface_components['unet'],
            diffusers_scheduler=huggingface_components['scheduler'],
            config=estimated_config
        )

        self.text_processing_engine = KolorsTextProcessingEngine(
            text_encoder=clip.cond_stage_model.clip_l,
            tokenizer=clip.tokenizer.clip_l,
            embedding_dir=dynamic_args['embedding_dir'],
            embedding_key='clip_l',
            embedding_expected_shape=768,
            emphasis_name=dynamic_args['emphasis_name'],
            text_projection=False,
            minimal_clip_skip=1,
            clip_skip=1,
            return_pooled=True,
            final_layer_norm=True,
        )

        self.embedder = Timestep(256)

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        # WebUI Legacy
        self.is_kolors = True

    def set_clip_skip(self, clip_skip):
        self.text_processing_engine.clip_skip = clip_skip

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str], skip_flag=True):
        if skip_flag:
            memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        # from ldm_patched.ldm.modules.encoders.noise_aug_modules import CLIPEmbeddingNoiseAugmentation
        # noise_augmentor = CLIPEmbeddingNoiseAugmentation(
        #     **{"noise_schedule_config": {"timesteps": 1100, "beta_schedule": "linear", "linear_start": 0.00085,
        #                                  "linear_end": 0.014}, "timestep_dim": 1280})

        cond_k, clip_pooled = self.text_processing_engine(prompt)

        width = getattr(prompt, 'width', 1024) or 1024
        height = getattr(prompt, 'height', 1024) or 1024
        is_negative_prompt = getattr(prompt, 'is_negative_prompt', False)

        crop_w = 0
        crop_h = 0
        target_width = width
        target_height = height

        out = [
            self.embedder(torch.Tensor([height])), self.embedder(torch.Tensor([width])),
            self.embedder(torch.Tensor([crop_h])), self.embedder(torch.Tensor([crop_w])),
            self.embedder(torch.Tensor([target_height])), self.embedder(torch.Tensor([target_width]))
        ]

        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1).to(clip_pooled)

        force_zero_negative_prompt = is_negative_prompt and all(x == '' for x in prompt)

        if force_zero_negative_prompt:
            clip_pooled = torch.zeros_like(clip_pooled)
            cond_k = torch.zeros_like(cond_k)

        cond = dict(
            crossattn=torch.cat([cond_k], dim=2),
            vector=torch.cat((clip_pooled.to(flat.device), flat), dim=1),
        )

        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        _, token_count = self.text_processing_engine.process_texts([prompt])
        return token_count, self.text_processing_engine.get_target_prompt_token_count(token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        sample = self.forge_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)
