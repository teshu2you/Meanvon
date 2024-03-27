#
# Harmonai.py
import gradio as gr
import os
from diffusers import DiffusionPipeline
import scipy.io.wavfile
import torch
import random
from resources.common import *

device_label_harmonai, model_arch = detect_device()
device_harmonai = torch.device(device_label_harmonai)

model_path_harmonai = "./models/harmonai/"
os.makedirs(model_path_harmonai, exist_ok=True)

model_list_harmonai = []

for filename in os.listdir(model_path_harmonai):
    f = os.path.join(model_path_harmonai, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors') or filename.endswith('.bin')):
        model_list_harmonai.append(f)

model_list_harmonai_builtin = [
    "harmonai/glitch-440k",
    "harmonai/honk-140k",
    "harmonai/jmann-small-190k",
    "harmonai/jmann-large-580k",
    "harmonai/maestro-150k",
    "harmonai/unlocked-250k",
]

for k in range(len(model_list_harmonai_builtin)):
    model_list_harmonai.append(model_list_harmonai_builtin[k])

@metrics_decoration
def music_harmonai(
    length_harmonai, 
    model_harmonai, 
    steps_harmonai, 
    seed_harmonai, 
    batch_size_harmonai, 
    batch_repeat_harmonai, 
    progress_harmonai=gr.Progress(track_tqdm=True)
    ):

    print(">>>[Harmonai 🔊 ]: starting module")

    if model_harmonai[0:9] == "./models/" :
        pipe_harmonai = DiffusionPipeline.from_single_file(model_harmonai, torch_dtype=torch.float32)
    else : 
        pipe_harmonai = DiffusionPipeline.from_pretrained(
            model_harmonai, 
            cache_dir=model_path_harmonai, 
            torch_dtype=model_arch,
            resume_download=True,
            local_files_only=True if offline_test() else None
            )
#    pipe_harmonai = pipe_harmonai.to(device_harmonai)
    if device_label_harmonai == "cuda" :
        pipe_harmonai.enable_sequential_cpu_offload()
    else : 
        pipe_harmonai = pipe_harmonai.to(device_harmonai)

    if seed_harmonai == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_harmonai
    generator = []
    for k in range(batch_repeat_harmonai):
        generator.append([torch.Generator(device_harmonai).manual_seed(final_seed + (k*batch_size_harmonai) + l ) for l in range(batch_size_harmonai)])

    final_seed = []
    for i in range (batch_repeat_harmonai):
        audios = pipe_harmonai(
            audio_length_in_s=length_harmonai,
            num_inference_steps=steps_harmonai,
            generator=generator[i],
            batch_size=batch_size_harmonai,
        ).audios

        for j, audio in enumerate(audios):
            seed_id = random_seed + i*batch_size_harmonai + j if (seed_harmonai == 0) else seed_harmonai + i*batch_size_harmonai + j
            savename = f"outputs/{seed_id}_{timestamper()}.wav"
            scipy.io.wavfile.write(savename, pipe_harmonai.unet.config.sample_rate, audio.transpose())
            final_seed.append(seed_id)

    print(f">>>[Harmonai 🔊 ]: generated {batch_repeat_harmonai} batch(es) of {batch_size_harmonai}")
    reporting_harmonai = f">>>[Harmonai 🔊 ]: "+\
        f"Settings : Model={model_harmonai} | "+\
        f"Steps={steps_harmonai} | "+\
        f"Duration={length_harmonai} sec. | "+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_harmonai) 
    
    del pipe_harmonai, generator, audios
    clean_ram()

    print(f">>>[Harmonai 🔊 ]: leaving module")
    return savename
