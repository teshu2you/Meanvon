#
# txt2vid_ze.py
import gradio as gr
import os
import imageio
from diffusers import TextToVideoZeroPipeline, TextToVideoZeroSDXLPipeline
import numpy as np
import torch
import random
from resources.common import *
from resources.gfpgan import *
import tomesd

device_label_txt2vid_ze, model_arch = detect_device()
device_txt2vid_ze = torch.device(device_label_txt2vid_ze)

model_path_txt2vid_ze = "./models/Stable_Diffusion/"
os.makedirs(model_path_txt2vid_ze, exist_ok=True)

model_list_txt2vid_ze = [
    "SG161222/Realistic_Vision_V3.0_VAE",
    "stabilityai/sdxl-turbo",
    "dataautogpt3/OpenDalleV1.1",
    "digiplay/AbsoluteReality_v1.8.1",
    "segmind/Segmind-Vega",
    "segmind/SSD-1B",
    "gsdf/Counterfeit-V2.5",
#    "ckpt/anything-v4.5-vae-swapped",
    "runwayml/stable-diffusion-v1-5",
    "nitrosocke/Ghibli-Diffusion",
]

# Bouton Cancel
stop_txt2vid_ze = False

def initiate_stop_txt2vid_ze() :
    global stop_txt2vid_ze
    stop_txt2vid_ze = True

def check_txt2vid_ze(step, timestep, latents) : 
    global stop_txt2vid_ze
    if stop_txt2vid_ze == False :
        return
    elif stop_txt2vid_ze == True :
        print(">>>[Text2Video-Zero 📼 ]: generation canceled by user")
        stop_txt2vid_ze = False
        try:
            del resources.txt2vid_ze.pipe_txt2vid_ze
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def video_txt2vid_ze(
    modelid_txt2vid_ze, 
    num_inference_step_txt2vid_ze, 
    sampler_txt2vid_ze, 
    guidance_scale_txt2vid_ze, 
    seed_txt2vid_ze, 
    num_frames_txt2vid_ze, 
    num_fps_txt2vid_ze, 
    height_txt2vid_ze, 
    width_txt2vid_ze, 
    num_videos_per_prompt_txt2vid_ze, 
    num_prompt_txt2vid_ze, 
    motion_field_strength_x_txt2vid_ze, 
    motion_field_strength_y_txt2vid_ze, 
    timestep_t0_txt2vid_ze :int, 
    timestep_t1_txt2vid_ze :int, 
    prompt_txt2vid_ze, 
    negative_prompt_txt2vid_ze, 
    nsfw_filter, 
    num_chunks_txt2vid_ze :int, 
    use_gfpgan_txt2vid_ze,
    tkme_txt2vid_ze,
    progress_txt2vid_ze=gr.Progress(track_tqdm=True)
    ):
    
    print(">>>[Text2Video-Zero 📼 ]: starting module")

    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_txt2vid_ze, device_txt2vid_ze, nsfw_filter)

    if (("XL" in modelid_txt2vid_ze.upper()) or (modelid_txt2vid_ze == "segmind/SSD-1B") or (modelid_txt2vid_ze == "segmind/Segmind-Vega") or (modelid_txt2vid_ze == "dataautogpt3/OpenDalleV1.1")) :
        is_xl_txt2vid_ze: bool = True
    else :        
        is_xl_txt2vid_ze: bool = False

    if (is_xl_txt2vid_ze == True):
        pipe_txt2vid_ze = TextToVideoZeroSDXLPipeline.from_pretrained(
            modelid_txt2vid_ze, 
            cache_dir=model_path_txt2vid_ze, 
            torch_dtype=model_arch, 
            use_safetensors=True, 
            safety_checker=nsfw_filter_final, 
            feature_extractor=feat_ex, 
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
    else:
        pipe_txt2vid_ze = TextToVideoZeroPipeline.from_pretrained(
            modelid_txt2vid_ze, 
            cache_dir=model_path_txt2vid_ze, 
            torch_dtype=model_arch, 
            use_safetensors=True, 
            safety_checker=nsfw_filter_final, 
            feature_extractor=feat_ex, 
            resume_download=True,
            local_files_only=True if offline_test() else None
        )

    pipe_txt2vid_ze = schedulerer(pipe_txt2vid_ze, sampler_txt2vid_ze)
    tomesd.apply_patch(pipe_txt2vid_ze, ratio=tkme_txt2vid_ze)
    if device_label_txt2vid_ze == "cuda" :
        pipe_txt2vid_ze.enable_sequential_cpu_offload()
    else : 
        pipe_txt2vid_ze = pipe_txt2vid_ze.to(device_txt2vid_ze)
#    pipe_txt2vid_ze.enable_vae_slicing()
    
    if seed_txt2vid_ze == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_txt2vid_ze
    generator = []
    for k in range(num_prompt_txt2vid_ze):
        generator.append(torch.Generator(device_txt2vid_ze).manual_seed(final_seed + k))

    final_seed = []
    for j in range (num_prompt_txt2vid_ze):
        if num_chunks_txt2vid_ze != 1 :
            result = []
            chunk_ids = np.arange(0, num_frames_txt2vid_ze, num_chunks_txt2vid_ze)
#            generator = torch.Generator(device=device_txt2vid_ze)
            for i in range(len(chunk_ids)):
                print(f"Processing chunk {i + 1} / {len(chunk_ids)}")
                ch_start = chunk_ids[i]
                ch_end = num_frames_txt2vid_ze if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
                if i == 0 :
                    frame_ids = [0] + list(range(ch_start, ch_end))
                else :
                    frame_ids = [ch_start -1] + list(range(ch_start, ch_end))
#                generator = generator.manual_seed(seed_txt2vid_ze)
                output = pipe_txt2vid_ze(
                    prompt=prompt_txt2vid_ze,
                    negative_prompt=negative_prompt_txt2vid_ze,
                    height=height_txt2vid_ze,
                    width=width_txt2vid_ze,
                    num_inference_steps=num_inference_step_txt2vid_ze,
                    guidance_scale=guidance_scale_txt2vid_ze,
                    frame_ids=frame_ids,
                    video_length=len(frame_ids), 
                    num_videos_per_prompt=num_videos_per_prompt_txt2vid_ze,
                    motion_field_strength_x=motion_field_strength_x_txt2vid_ze,
                    motion_field_strength_y=motion_field_strength_y_txt2vid_ze,
                    t0=timestep_t0_txt2vid_ze,
                    t1=timestep_t1_txt2vid_ze,
                    generator = generator[j],
                    callback=check_txt2vid_ze, 
                )
                result.append(output.images[1:])
            result = np.concatenate(result)
        else :
            result = pipe_txt2vid_ze(
                prompt=prompt_txt2vid_ze,
                negative_prompt=negative_prompt_txt2vid_ze,
                height=height_txt2vid_ze,
                width=width_txt2vid_ze,
                num_inference_steps=num_inference_step_txt2vid_ze,
                guidance_scale=guidance_scale_txt2vid_ze,
                video_length=num_frames_txt2vid_ze,
                num_videos_per_prompt=num_videos_per_prompt_txt2vid_ze,
                motion_field_strength_x=motion_field_strength_x_txt2vid_ze,
                motion_field_strength_y=motion_field_strength_y_txt2vid_ze,
                t0=timestep_t0_txt2vid_ze,
                t1=timestep_t1_txt2vid_ze,
                generator = generator[j],
                callback=check_txt2vid_ze, 
            ).images
         
        result = [(r * 255).astype("uint8") for r in result]

        for n in range(len(result)):
            if use_gfpgan_txt2vid_ze == True :
                result[n] = image_gfpgan_mini(result[n])

        a = 1
        b = 0
        for o in range(len(result)):
            if (a < num_frames_txt2vid_ze):
                a += 1
            elif (a == num_frames_txt2vid_ze):
                seed_id = random_seed + j*num_videos_per_prompt_txt2vid_ze + b if (seed_txt2vid_ze == 0) else seed_txt2vid_ze + j*num_videos_per_prompt_txt2vid_ze + b
                savename = f"outputs/{seed_id}_{timestamper()}.mp4"
                imageio.mimsave(savename, result, fps=num_fps_txt2vid_ze)
                final_seed.append(seed_id)
                a = 1
                b += 1

    print(f">>>[Text2Video-Zero 📼 ]: generated {num_prompt_txt2vid_ze} batch(es) of {num_videos_per_prompt_txt2vid_ze}")
    reporting_txt2vid_ze = f">>>[Text2Video-Zero 📼 ]: "+\
        f"Settings : Model={modelid_txt2vid_ze} | "+\
        f"Sampler={sampler_txt2vid_ze} | "+\
        f"Steps={num_inference_step_txt2vid_ze} | "+\
        f"CFG scale={guidance_scale_txt2vid_ze} | "+\
        f"Video length={num_frames_txt2vid_ze} frames | "+\
        f"FPS={num_fps_txt2vid_ze} frames | "+\
        f"Chunck size={num_chunks_txt2vid_ze} | "+\
        f"Size={width_txt2vid_ze}x{height_txt2vid_ze} | "+\
        f"Motion field strength x={motion_field_strength_x_txt2vid_ze} | "+\
        f"Motion field strength y={motion_field_strength_y_txt2vid_ze} | "+\
        f"Timestep t0={timestep_t0_txt2vid_ze} | "+\
        f"Timestep t1={timestep_t1_txt2vid_ze} | "+\
        f"GFPGAN={use_gfpgan_txt2vid_ze} | "+\
        f"Token merging={tkme_txt2vid_ze} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Prompt={prompt_txt2vid_ze} | "+\
        f"Negative prompt={negative_prompt_txt2vid_ze} | "+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_txt2vid_ze) 

    del nsfw_filter_final, feat_ex, pipe_txt2vid_ze, generator, result
    clean_ram()

    print(f">>>[Text2Video-Zero 📼 ]: leaving module")
    return savename
