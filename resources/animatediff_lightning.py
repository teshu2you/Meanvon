#
import gradio as gr
import os
import imageio
from diffusers.utils import export_to_video, export_to_gif
import numpy as np
import torch
import random
from resources.common import *
from resources.gfpgan import *
import tomesd
from modules.util import free_cuda_mem, free_cuda_cache
import traceback
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

os.makedirs("./outputs/animatediff_lightning/", exist_ok=True)

model_path_animatediff_lightning = "./models/Animate/checkpoints/"
os.makedirs(model_path_animatediff_lightning, exist_ok=True)

adapter_path_animatediff_lightning = "./models/Animate/AnimateLightning/adapter/"
os.makedirs(adapter_path_animatediff_lightning, exist_ok=True)

lora_path_animatediff_lightning = "./models/Animate/AnimateLightning/loras/"
os.makedirs(lora_path_animatediff_lightning, exist_ok=True)

model_list_animatediff_lightning_builtin = [
    "emilianJR/epiCRealism",
    "SG161222/Realistic_Vision_V3.0_VAE",
    "gsdf/Counterfeit-V2.5",
    "ckpt/anything-v4.5-vae-swapped"
]

adapter_list_animatediff_lightning_builtin = [
    "ByteDance/AnimateDiff-Lightning",
]

# lora_list_animatediff_lightning_builtin = []


def get_obj_list(path, builtin):
    obj_list = []
    for file_path, _, file_list in os.walk(path):
        for filename in file_list:
            f = os.path.join(file_path, filename)
            if os.path.isfile(f) and (
                    filename.endswith('.ckpt') or filename.endswith('.bin') or filename.endswith('.safetensors')):
                if "models--" in file_path:
                    continue
                obj_list.append(os.path.dirname(f))

    for k in range(len(builtin)):
        obj_list.append(builtin[k])
    return obj_list


model_list_animatediff_lightning = get_obj_list(path=model_path_animatediff_lightning,
                                                builtin=model_list_animatediff_lightning_builtin)
adapter_list_animatediff_lightning = get_obj_list(path=adapter_path_animatediff_lightning,
                                                  builtin=adapter_list_animatediff_lightning_builtin)
# lora_list_animatediff_lightning = get_obj_list(path=lora_path_animatediff_lightning,
#                                                builtin=lora_list_animatediff_lightning_builtin)

# Bouton Cancel
stop_animatediff_lightning = False


def initiate_stop_animatediff_lightning():
    global stop_animatediff_lightning
    stop_animatediff_lightning = True


def check_animatediff_lightning(pipe, step_index, timestep, callback_kwargs):
    global stop_animatediff_lightning
    if stop_animatediff_lightning:
        print(">>>[Animate lightning 📼 ]: generation canceled by user")
        stop_animatediff_lightning = False
        pipe._interrupt = True
    return callback_kwargs


@metrics_decoration
def video_animatediff_lightning(
        modelid_animatediff_lightning,
        adapterid_animatediff_lightning,
        # loraid_animatediff_lightning,
        num_inference_step_animatediff_lightning,
        sampler_animatediff_lightning,
        guidance_scale_animatediff_lightning,
        seed_animatediff_lightning,
        num_frames_animatediff_lightning,
        height_animatediff_lightning,
        width_animatediff_lightning,
        num_videos_per_prompt_animatediff_lightning,
        num_prompt_animatediff_lightning,
        prompt_animatediff_lightning,
        negative_prompt_animatediff_lightning,
        nsfw_filter,
        use_gfpgan_animatediff_lightning,
        tkme_animatediff_lightning,
        progress_animatediff_lightning=gr.Progress(track_tqdm=True)
):
    try:
        print(">>>[Animate lightning 📼 ]: starting module")
        device_label_animatediff_lightning, model_arch = detect_device()
        print(f"[device_label_animatediff_lightning]: {device_label_animatediff_lightning}  |   [arch]: {model_arch}")
        device_animatediff_lightning = torch.device(device_label_animatediff_lightning)

        nsfw_filter_final, feat_ex = safety_checker_sd(model_path_animatediff_lightning, device_animatediff_lightning,
                                                       nsfw_filter)

        adapter_animatediff_lightning = MotionAdapter.from_pretrained(
            adapterid_animatediff_lightning,
            cache_dir=adapter_path_animatediff_lightning,
            torch_dtype=model_arch,
            use_safetensors=True,
            resume_download=True,
            local_files_only=False
        ).to(device=device_label_animatediff_lightning, dtype=model_arch)

        # adapter_animatediff_lightning.load_state_dict(
        #     load_file(modelid_animatediff_lightning, device=device_label_animatediff_lightning))

        if modelid_animatediff_lightning in model_list_animatediff_lightning:
            pipe_animatediff_lightning = AnimateDiffPipeline.from_pretrained(
                modelid_animatediff_lightning,
                cache_dir=model_path_animatediff_lightning,
                torch_dtype=model_arch,
                motion_adapter=adapter_animatediff_lightning,
                use_safetensors=True,
                safety_checker=nsfw_filter_final,
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=False
            )
        else:
            pipe_animatediff_lightning = modelid_animatediff_lightning

        # pipe_animatediff_lightning.load_lora_weights(
        #     loraid_animatediff_lightning,
        #     weight_name="sdxl_lightning_8step_lora.safetensors",
        #     cache_dir=lora_path_animatediff_lightning,
        #     use_safetensors=True,
        #     adapter_name="adapter1",
        #     low_cpu_mem_usage=False,
        #     ignore_mismatched_sizes=True,
        #     resume_download=True,
        #     local_files_only=False
        # )

        pipe_animatediff_lightning.scheduler = EulerDiscreteScheduler.from_config(
            pipe_animatediff_lightning.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

        if device_label_animatediff_lightning == "cuda":
            pipe_animatediff_lightning.enable_sequential_cpu_offload()
        else:
            pipe_animatediff_lightning = pipe_animatediff_lightning.to(device_animatediff_lightning)
        pipe_animatediff_lightning.enable_vae_slicing()

        if seed_animatediff_lightning == 0:
            random_seed = random.randrange(0, 10000000000, 1)
            final_seed = random_seed
        else:
            final_seed = seed_animatediff_lightning
        generator = []
        for k in range(num_prompt_animatediff_lightning):
            generator.append(torch.Generator(device_animatediff_lightning).manual_seed(final_seed + k))

        result = None
        final_seed = []
        for i in range(num_prompt_animatediff_lightning):
            result = pipe_animatediff_lightning(
                prompt=prompt_animatediff_lightning,
                negative_prompt=negative_prompt_animatediff_lightning,
                num_frames=num_frames_animatediff_lightning,
                height=height_animatediff_lightning,
                width=width_animatediff_lightning,
                num_inference_steps=num_inference_step_animatediff_lightning,
                guidance_scale=guidance_scale_animatediff_lightning,
                video_length=num_frames_animatediff_lightning,
                num_videos_per_prompt=num_videos_per_prompt_animatediff_lightning,
                generator=generator[i],
                callback_on_step_end=check_animatediff_lightning,
                callback_on_step_end_tensor_inputs=['latents'],
            ).frames[0]

            timestamp = time.time()
            seed_id = random_seed + i * num_videos_per_prompt_animatediff_lightning if (
                    seed_animatediff_lightning == 0) else seed_animatediff_lightning + i * num_videos_per_prompt_animatediff_lightning
            savename = f"outputs/animatediff_lightning/{seed_id}_{timestamper()}.mp4"
            export_to_video(result, savename, fps=8)
            final_seed.append(seed_id)

        print(
            f">>>[Animate_lightning 📼 ]: generated {num_prompt_animatediff_lightning} batch(es) of {num_videos_per_prompt_animatediff_lightning}")
        reporting_animatediff_lightning = f">>>[Animate lightning 📼 ]: " + \
                                          f"Settings : Model={modelid_animatediff_lightning} | " + \
                                          f"Sampler={sampler_animatediff_lightning} | " + \
                                          f"Steps={num_inference_step_animatediff_lightning} | " + \
                                          f"CFG scale={guidance_scale_animatediff_lightning} | " + \
                                          f"Video length={num_frames_animatediff_lightning} frames | " + \
                                          f"Size={width_animatediff_lightning}x{height_animatediff_lightning} | " + \
                                          f"GFPGAN={use_gfpgan_animatediff_lightning} | " + \
                                          f"Token merging={tkme_animatediff_lightning} | " + \
                                          f"nsfw_filter={bool(int(nsfw_filter))} | " + \
                                          f"Prompt={prompt_animatediff_lightning} | " + \
                                          f"Negative prompt={negative_prompt_animatediff_lightning} | " + \
                                          f"Seed List=" + ', '.join(
            [f"{final_seed[m]}" for m in range(len(final_seed))])
        print(reporting_animatediff_lightning)
    except ConnectionResetError as e:
        print("[Error]: {}".format(e))
        pass
    except:
        traceback.print_exc()
    finally:
        del nsfw_filter_final, feat_ex, pipe_animatediff_lightning, generator, result
        clean_ram()

        print(f">>>[Animate lightning 📼 ]: leaving module")
        return savename
