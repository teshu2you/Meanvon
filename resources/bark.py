#
# bark.py
import gradio as gr
import os
import traceback
from optimum.bettertransformer import BetterTransformer
from transformers import AutoProcessor, BarkModel
from scipy.io.wavfile import write as write_wav
import random
from resources.common import *
from modules.util import free_cuda_mem, free_cuda_cache

model_path_bark = "./models/Bark/"
os.makedirs(model_path_bark, exist_ok=True)

model_list_bark = [
    "suno/bark-small",
    "suno/bark",
]

# https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c
voice_preset_list_bark = {
    "German Male": "v2/de_speaker_9",
    "German Female": "v2/de_speaker_8",
    "English Male": "v2/en_speaker_6",
    "English Female": "v2/en_speaker_9",
    "Spanish Male": "v2/es_speaker_7",
    "Spanish Female": "v2/es_speaker_9",
    "French Male": "v2/fr_speaker_8",
    "French Female": "v2/fr_speaker_5",
    "Hindi Male": "v2/hi_speaker_8",
    "Hindi Female": "v2/hi_speaker_9",
    "Japanese Male": "v2/ja_speaker_6",
    "Japanese Female": "v2/ja_speaker_7",
    "Korean Male": "v2/ko_speaker_9",
    "Korean Female": "v2/ko_speaker_0",
    "Polish Male": "v2/pl_speaker_8",
    "Polish Female": "v2/pl_speaker_9",
    "Portuguese Male": "v2/pt_speaker_9",
    "Russian Male": "v2/ru_speaker_7",
    "Russian Female": "v2/ru_speaker_5",
    "Turkish Male": "v2/tr_speaker_9",
    "Turkish Female": "v2/tr_speaker_5",
    "Chinese, simplified Male": "v2/zh_speaker_8",
    "Chinese, simplified Female": "v2/zh_speaker_9",
}


@metrics_decoration
def music_bark(
        prompt_bark,
        model_bark,
        voice_preset_bark,
        progress_bark=gr.Progress(track_tqdm=True)
):
    try:
        print(">>>[Bark 🗣️ ]: starting module")
        savename = f"outputs/bark/{timestamper()}.wav"
        free_cuda_mem()
        device_label_bark, model_arch = detect_device()
        print(f"[device_label_bark]: {device_label_bark}  |   [arch]: {model_arch}")
        device_bark = torch.device(device_label_bark)

        if "small" in model_bark:
            device_label_bark = device_bark = "cpu"
            model_arch = torch.float32
        print(f"[NEW --> device_label_bark]: {device_label_bark}  |   [arch]: {model_arch}")

        processor = AutoProcessor.from_pretrained(
            model_bark,
            cache_dir=model_path_bark,
            torch_dtype=model_arch,
            resume_download=True,
            local_files_only=False
        )

        pipe_bark = BarkModel.from_pretrained(
            model_bark,
            cache_dir=model_path_bark,
            torch_dtype=model_arch,
            resume_download=True,
            local_files_only=False
        )

        if model_arch == torch.float16:
            pipe_bark = pipe_bark.to(device_bark)
            pipe_bark = BetterTransformer.transform(pipe_bark, keep_original_model=False)
            if device_label_bark == "cuda":
                pipe_bark.enable_cpu_offload()

        voice_preset = voice_preset_list_bark[voice_preset_bark]
        inputs = processor(prompt_bark, voice_preset=voice_preset)
        audio_array = pipe_bark.generate(**inputs, do_sample=True)
        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = pipe_bark.generation_config.sample_rate

        write_wav(savename, sample_rate, audio_array)

        print(f">>>[Bark 🗣️ ]: generated 1 audio file")
        reporting_bark = f">>>[Bark 🗣️ ]: " + \
                         f"Settings : Model={model_bark} | " + \
                         f"Voice preset={voice_preset_bark} | " + \
                         f"Prompt={prompt_bark}"
        print(reporting_bark)

    except:
        traceback.print_exc()
    finally:
        del processor, pipe_bark, audio_array
        clean_ram()

        print(f">>>[Bark 🗣️ ]: leaving module")
        return savename
