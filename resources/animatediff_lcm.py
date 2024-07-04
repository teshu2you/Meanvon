#
# animatediff_lcm.py
import gradio as gr
import os
import imageio
from diffusers import AnimateDiffPipeline, MotionAdapter
from diffusers.utils import export_to_video, export_to_gif
import numpy as np
from compel import Compel, ReturnedEmbeddingsType
import torch
import random
from resources.common import *
from resources.gfpgan import *
import tomesd
from modules.util import free_cuda_mem, free_cuda_cache
import traceback

model_path_animatediff_lcm = "./models/Animate/checkpoints/"
os.makedirs(model_path_animatediff_lcm, exist_ok=True)

adapter_path_animatediff_lcm = "./models/Animate/AnimateLCM/adapter/"
os.makedirs(adapter_path_animatediff_lcm, exist_ok=True)

lora_path_animatediff_lcm = "./models/Animate/AnimateLCM/loras/"
os.makedirs(lora_path_animatediff_lcm, exist_ok=True)

model_list_animatediff_lcm_builtin = [
    "emilianJR/epiCRealism",
    "SG161222/Realistic_Vision_V3.0_VAE",
    "stabilityai/sdxl-turbo",
    "dataautogpt3/OpenDalleV1.1",
    "digiplay/AbsoluteReality_v1.8.1",
    "segmind/Segmind-Vega",
    "segmind/SSD-1B",
    "gsdf/Counterfeit-V2.5",
    "ckpt/anything-v4.5-vae-swapped",
    "runwayml/stable-diffusion-v1-5",
    "nitrosocke/Ghibli-Diffusion",
]

model_list_adapters_animatediff_lcm = {
    "wangfuyun/AnimateLCM":("AnimateLCM_sd15_t2v_lora.safetensors", 0.8),
    "ByteDance/AnimateDiff-Lightning":("animatediff_lightning_4step_diffusers.safetensors", 1.0),
}

adapter_list_animatediff_lcm_builtin = [
    "wangfuyun/AnimateLCM",
    "wangfuyun/AnimateLCM-I2V",
    "wangfuyun/AnimateLCM-SVD-xt"
]

lora_list_animatediff_lcm_builtin = []


def get_obj_list(path, builtin):
    obj_list = []
    for file_path, _, file_list in os.walk(path):
        for filename in file_list:
            f = os.path.join(file_path, filename)
            if os.path.isfile(f) and (
                    filename.endswith('.ckpt') or filename.endswith('.bin') or filename.endswith('.safetensors')):
                # if any([e in filename.lower() for e in ['_lcm', '_turbo']]):
                #     lora_list_animatediff_lcm.append(os.path.dirname(f))
                if "models--" in file_path:
                    continue
                obj_list.append(os.path.dirname(f))

    for k in range(len(builtin)):
        obj_list.append(builtin[k])
    return obj_list


model_list_animatediff_lcm = get_obj_list(path=model_path_animatediff_lcm, builtin=model_list_animatediff_lcm_builtin)
adapter_list_animatediff_lcm = get_obj_list(path=adapter_path_animatediff_lcm,
                                            builtin=adapter_list_animatediff_lcm_builtin)
lora_list_animatediff_lcm = get_obj_list(path=lora_path_animatediff_lcm, builtin=lora_list_animatediff_lcm_builtin)

# Bouton Cancel
stop_animatediff_lcm = False


def initiate_stop_animatediff_lcm():
    global stop_animatediff_lcm
    stop_animatediff_lcm = True


def check_animatediff_lcm(pipe, step_index, timestep, callback_kwargs):
    global stop_animatediff_lcm
    if stop_animatediff_lcm:
        print(">>>[AnimateLCM 📼 ]: generation canceled by user")
        stop_animatediff_lcm = False
        pipe._interrupt = True
    return callback_kwargs


@metrics_decoration
def video_animatediff_lcm(
        modelid_animatediff_lcm,
        adapterid_animatediff_lcm,
        loraid_animatediff_lcm,
        num_inference_step_animatediff_lcm,
        sampler_animatediff_lcm,
        guidance_scale_animatediff_lcm,
        seed_animatediff_lcm,
        num_frames_animatediff_lcm,
        num_fps_animatediff_lcm,
        height_animatediff_lcm,
        width_animatediff_lcm,
        num_videos_per_prompt_animatediff_lcm,
        num_prompt_animatediff_lcm,
        prompt_animatediff_lcm,
        negative_prompt_animatediff_lcm,
        output_type_animatediff_lcm,
        nsfw_filter,
        use_gfpgan_animatediff_lcm,
        tkme_animatediff_lcm,
        progress_animatediff_lcm=gr.Progress(track_tqdm=True)
):
    try:
        print(">>>[AnimateLCM 📼 ]: starting module")
        device_label_animatediff_lcm, model_arch = detect_device()
        print(f"[device_label_animatediff_lcm]: {device_label_animatediff_lcm}  |   [arch]: {model_arch}")
        device_animatediff_lcm = torch.device(device_label_animatediff_lcm)

        nsfw_filter_final, feat_ex = safety_checker_sd(model_path_animatediff_lcm, device_animatediff_lcm, nsfw_filter)

        adapter_animatediff_lcm = MotionAdapter.from_pretrained(
            adapterid_animatediff_lcm,
            cache_dir=adapter_path_animatediff_lcm,
            torch_dtype=model_arch,
            use_safetensors=True,
            resume_download=True,
            local_files_only=False
        )

        if modelid_animatediff_lcm in model_list_animatediff_lcm:
            pipe_animatediff_lcm = AnimateDiffPipeline.from_pretrained(
                modelid_animatediff_lcm,
                cache_dir=model_path_animatediff_lcm,
                torch_dtype=model_arch,
                motion_adapter=adapter_animatediff_lcm,
                use_safetensors=True,
                safety_checker=nsfw_filter_final,
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=False
            )
        else:
            pipe_animatediff_lcm = modelid_animatediff_lcm

        pipe_animatediff_lcm.load_lora_weights(
            loraid_animatediff_lcm,
            weight_name="sd15_lcm_lora_beta.safetensors",
            cache_dir=lora_path_animatediff_lcm,
            use_safetensors=True,
            adapter_name="adapter1",
            resume_download=True,
            local_files_only=False
        )

        pipe_animatediff_lcm.fuse_lora(lora_scale=0.8)
        #    pipe_animatediff_lcm.fuse_lora(lora_scale=lora_weight_animatediff_lcm)
        #    pipe_animatediff_lcm.set_adapters(["adapter1"], adapter_weights=[float(lora_weight_animatediff_lcm)])

        pipe_animatediff_lcm.scheduler = LCMScheduler.from_config(pipe_animatediff_lcm.scheduler.config,
                                                                  beta_schedule="linear")
        #    pipe_animatediff_lcm = schedulerer(pipe_animatediff_lcm, sampler_animatediff_lcm)
        #    tomesd.apply_patch(pipe_animatediff_lcm, ratio=tkme_animatediff_lcm)
        if device_label_animatediff_lcm == "cuda":
            pipe_animatediff_lcm.enable_sequential_cpu_offload()
        else:
            pipe_animatediff_lcm = pipe_animatediff_lcm.to(device_animatediff_lcm)
        pipe_animatediff_lcm.enable_vae_slicing()
        # pipe_animatediff_lcm.enable_model_cpu_offload()

        if seed_animatediff_lcm == 0:
            random_seed = random.randrange(0, 10000000000, 1)
            final_seed = random_seed
        else:
            final_seed = seed_animatediff_lcm
        generator = []
        for k in range(num_prompt_animatediff_lcm):
            generator.append(torch.Generator(device_animatediff_lcm).manual_seed(final_seed + k))

        prompt_animatediff_lcm = str(prompt_animatediff_lcm)
        negative_prompt_animatediff_lcm = str(negative_prompt_animatediff_lcm)
        if prompt_animatediff_lcm == "None":
            prompt_animatediff_lcm = ""
        if negative_prompt_animatediff_lcm == "None":
            negative_prompt_animatediff_lcm = ""

        compel = Compel(tokenizer=pipe_animatediff_lcm.tokenizer, text_encoder=pipe_animatediff_lcm.text_encoder,
                        truncate_long_prompts=False, device=device_animatediff_lcm)
        conditioning = compel.build_conditioning_tensor(prompt_animatediff_lcm)
        neg_conditioning = compel.build_conditioning_tensor(negative_prompt_animatediff_lcm)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length(
            [conditioning, neg_conditioning])

        result = None
        if output_type_animatediff_lcm == "gif":
            savename_final = []
        final_seed = []
        for i in range(num_prompt_animatediff_lcm):
            result = pipe_animatediff_lcm(
                prompt_embeds=conditioning,
                negative_prompt_embeds=neg_conditioning,
                # prompt=prompt_animatediff_lcm,
                # negative_prompt=negative_prompt_animatediff_lcm,
                num_frames=num_frames_animatediff_lcm,
                height=height_animatediff_lcm,
                width=width_animatediff_lcm,
                num_inference_steps=num_inference_step_animatediff_lcm,
                guidance_scale=guidance_scale_animatediff_lcm,
                video_length=num_frames_animatediff_lcm,
                num_videos_per_prompt=num_videos_per_prompt_animatediff_lcm,
                generator=generator[i],
                callback_on_step_end=check_animatediff_lcm,
                callback_on_step_end_tensor_inputs=['latents'],
            ).frames[0]

            timestamp = time.time()
            seed_id = random_seed + i * num_videos_per_prompt_animatediff_lcm if (
                    seed_animatediff_lcm == 0) else seed_animatediff_lcm + i * num_videos_per_prompt_animatediff_lcm
            if output_type_animatediff_lcm == "mp4":
                savename = "outputs/tmp_animatelcm_out.mp4"
                savename_final = name_seeded_video(seed_id)
                export_to_video(result, savename, fps=num_fps_animatediff_lcm)
                os.rename(savename, savename_final)
            elif output_type_animatediff_lcm == "gif":
                savename_final = []
                savename = "outputs/tmp_animatelcm_out.gif"
                savename_rename = name_seeded_gif(seed_id)
                export_to_gif(result, savename, fps=num_fps_animatediff_lcm)
                os.rename(savename, savename_rename)
                savename_final.append(savename_rename)
            final_seed.append(seed_id)

        print(
            f">>>[AnimateLCM 📼 ]: generated {num_prompt_animatediff_lcm} batch(es) of {num_videos_per_prompt_animatediff_lcm}")
        reporting_animatediff_lcm = f">>>[AnimateLCM 📼 ]: " + \
                                    f"Settings : Model={modelid_animatediff_lcm} | " + \
                                    f"Sampler={sampler_animatediff_lcm} | " + \
                                    f"Steps={num_inference_step_animatediff_lcm} | " + \
                                    f"CFG scale={guidance_scale_animatediff_lcm} | " + \
                                    f"Video length={num_frames_animatediff_lcm} frames | " + \
                                    f"Size={width_animatediff_lcm}x{height_animatediff_lcm} | " + \
                                    f"GFPGAN={use_gfpgan_animatediff_lcm} | " + \
                                    f"Token merging={tkme_animatediff_lcm} | " + \
                                    f"nsfw_filter={bool(int(nsfw_filter))} | " + \
                                    f"Prompt={prompt_animatediff_lcm} | " + \
                                    f"Negative prompt={negative_prompt_animatediff_lcm} | " + \
                                    f"Seed List=" + ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
        print(reporting_animatediff_lcm)

    except:
        traceback.print_exc()
    finally:
        del nsfw_filter_final, feat_ex, pipe_animatediff_lcm, generator, result
        clean_ram()

        print(f">>>[AnimateLCM 📼 ]: leaving module")
        return savename
