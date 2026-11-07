"""
Copyright (C) 2024 lllyasviel

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/
"""

from .detection import model_config_from_unet, unet_prefix_from_state_dict

# FLUX.3（f3-image-small 系列）的独有键：content/txt 双流 block + 早期调制。
# 本仓库没有对应实现，这里显式拦截，避免一路抛只写 "Failed to recognize model..."
# 的模糊错误（上层会顺带按 Flux1 配错 TE/VAE）。
_FLUX3_KEY_PREFIXES = (
    "content_mode_blocks.",
    "txt_mode_blocks.",
    "early_stream_modulations.",
    "emb_in.image",
)


def looks_like_flux3(state_dict) -> bool:
    try:
        keys = state_dict.keys()
    except AttributeError:
        return False
    for k in keys:
        if isinstance(k, str) and k.startswith(_FLUX3_KEY_PREFIXES):
            return True
    return False


def guess(state_dict):
    unet_key_prefix = unet_prefix_from_state_dict(state_dict)
    if (result := model_config_from_unet(state_dict, unet_key_prefix, use_base_if_no_match=False)) is None:
        if looks_like_flux3(state_dict):
            raise ModuleNotFoundError(
                "识别到 FLUX.3（f3-image-small）布局，但没能匹配到模型配置。"
                "请确认 modules_forge/packages/huggingface_guess/model_list.py 里有 Flux3Image，"
                "且 detection.py 的 flux3 分支能从 emb_in.image.weight 推出 hidden_size/num_heads。")
        raise ModuleNotFoundError("Failed to recognize model...")
    result.unet_key_prefix = [unet_key_prefix]
    result.unet_config.pop("image_model", None)
    result.unet_config.pop("audio_model", None)
    return result


def guess_repo_name(state_dict):
    config = guess(state_dict)
    assert config is not None
    repo_id = config.huggingface_repo
    return repo_id
