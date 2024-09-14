import torch
import ldm_patched.modules.model_management
import ldm_patched.modules.utils
import folder_paths
import os
import logging

CLAMP_QUANTILE = 0.99

def extract_lora(diff, rank):
    conv2d = (len(diff.shape) == 4)
    kernel_size = None if not conv2d else diff.size()[2:4]
    conv2d_3x3 = conv2d and kernel_size != (1, 1)
    out_dim, in_dim = diff.size()[0:2]
    rank = min(rank, in_dim, out_dim)

    if conv2d:
        if conv2d_3x3:
            diff = diff.flatten(start_dim=1)
        else:
            diff = diff.squeeze()


    U, S, Vh = torch.linalg.svd(diff.float())
    U = U[:, :rank]
    S = S[:rank]
    U = U @ torch.diag(S)
    Vh = Vh[:rank, :]

    dist = torch.cat([U.flatten(), Vh.flatten()])
    hi_val = torch.quantile(dist, CLAMP_QUANTILE)
    low_val = -hi_val

    U = U.clamp(low_val, hi_val)
    Vh = Vh.clamp(low_val, hi_val)
    if conv2d:
        U = U.reshape(out_dim, rank, 1, 1)
        Vh = Vh.reshape(rank, in_dim, kernel_size[0], kernel_size[1])
    return (U, Vh)

class LoraSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"filename_prefix": ("STRING", {"default": "loras/ComfyUI_extracted_lora"}),
                              "rank": ("INT", {"default": 8, "min": 1, "max": 1024, "step": 1}),
                            },
                "optional": {"model_diff": ("MODEL",),},
    }
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "_for_testing"

    def save(self, filename_prefix, rank, model_diff=None):
        if model_diff is None:
            return {}

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        output_sd = {}
        prefix_key = "diffusion_model."
        stored = set()

        comfy.model_management.load_models_gpu([model_diff], force_patch_weights=True)
        sd = model_diff.model_state_dict(filter_prefix=prefix_key)

        for k in sd:
            if k.endswith(".weight"):
                weight_diff = sd[k]
                if weight_diff.ndim < 2:
                    continue
                try:
                    out = extract_lora(weight_diff, rank)
                    output_sd["{}.lora_up.weight".format(k[:-7])] = out[0].contiguous().half().cpu()
                    output_sd["{}.lora_down.weight".format(k[:-7])] = out[1].contiguous().half().cpu()
                except:
                    logging.warning("Could not generate lora weights for key {}, is the weight difference a zero?".format(k))

        output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

        comfy.utils.save_torch_file(output_sd, output_checkpoint, metadata=None)
        return {}

NODE_CLASS_MAPPINGS = {
    "LoraSave": LoraSave
}
