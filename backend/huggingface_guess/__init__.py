from backend.huggingface_guess.detection import model_config_from_unet, unet_prefix_from_state_dict, model_config_from_diffusers_unet

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
    result = model_config_from_unet(state_dict, unet_key_prefix, use_base_if_no_match=False)
    if result is None:
        result = model_config_from_diffusers_unet(state_dict)
    result.unet_key_prefix = [unet_key_prefix]
    if 'image_model' in result.unet_config:
        del result.unet_config['image_model']
    if 'audio_model' in result.unet_config:
        del result.unet_config['audio_model']
    return result


def guess_repo_name(state_dict):
    config = guess(state_dict)
    assert config is not None
    repo_id = config.huggingface_repo
    return repo_id
