import torch

from modules.util import load_model
from modules.config import svd_config
from modules.util import show_cuda_info, free_cuda_mem, free_cuda_cache

def load_video_module(model_load_flag):
    version = svd_config.get("version")  # @param ["modules", "svd_xt"]

    if version in svd_config.keys():
        num_frames = svd_config.get(version).get("num_frames")
        num_steps = svd_config.get(version).get("num_steps")
        # output_folder = default(output_folder, "outputs/simple_video_sample/modules/")
        model_config = svd_config.get(version).get("model_config")
    else:
        raise ValueError(f"Version {version} does not exist.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None

    if not model_load_flag:
        return device, model

    try:
        print(f"[load_video_module]: use {device}")
        model = load_model(
            model_config,
            device,
            num_frames,
            num_steps
        )
    except Exception as e:
        print(f"[load_video_module]: ERROR-> {e.__str__()}")
        free_cuda_mem()
        free_cuda_cache()
        if "CUDA out of memory" in e.__str__():
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            print(f"[load_video_module]: use {device}")
            model = load_model(
                model_config,
                device,
                num_frames,
                num_steps
            )
    return device, model
