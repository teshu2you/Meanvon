# Attention !
import os
import torch
import logging
import importlib

import backend.args
import backend.patcher_diffusion
from diffusers import DiffusionPipeline
from transformers import modeling_utils

# from backend import memory_management
from backend.utils import read_arbitrary_config, load_torch_file, beautiful_print_gguf_state_dict_statics
from backend.state_dict import try_filter_state_dict, load_state_dict
from backend.operations import using_forge_operations
from backend.nn.vae import IntegratedAutoencoderKL
from backend.nn.clip import IntegratedCLIP, IntegratedCLIPKolors
from backend.nn.unet import IntegratedUNet2DConditionModel
from backend.nn.kolors import IntegratedKolors2DConditionModel
from backend import huggingface_guess, memory_management
from backend.diffusion_engine.sd15 import StableDiffusion
from backend.diffusion_engine.sd20 import StableDiffusion2
from backend.diffusion_engine.sdxl import StableDiffusionXL
from backend.diffusion_engine.kolors import Kolors
from backend.diffusion_engine.flux import Flux
from util.printf import printF, MasterName
from backend.patcher_diffusion.kolors.models.modeling_chatglm import ChatGLMModel, ChatGLMConfig
from backend.patcher_diffusion.kolors.models.tokenization_chatglm import ChatGLMTokenizer

possible_models = [StableDiffusion, StableDiffusion2, StableDiffusionXL, Flux, Kolors]


# logging.getLogger("diffusers").setLevel(logging.ERROR)
dir_path = os.path.dirname(__file__)


def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
    config_path = os.path.join(repo_path, component_name)

    if component_name in ['feature_extractor', 'safety_checker']:
        return None

    if lib_name in ['transformers', 'diffusers', 'kolors']:
        if component_name in ['scheduler']:
            cls = getattr(importlib.import_module(lib_name), cls_name)
            return cls.from_pretrained(os.path.join(repo_path, component_name))
        if component_name.startswith('tokenizer'):
            if cls_name == 'ChatGLMTokenizer':
                comp = ChatGLMTokenizer.from_pretrained(os.path.join(repo_path, component_name))
            else:
                cls = getattr(importlib.import_module(lib_name), cls_name)
                comp = cls.from_pretrained(os.path.join(repo_path, component_name))
                comp._eventual_warn_about_too_long_sequence = lambda *args, **kwargs: None

            return comp

        if cls_name in ['AutoencoderKL']:
            assert isinstance(state_dict, dict) and len(state_dict) > 16, 'You do not have VAE state dict!'

            config = IntegratedAutoencoderKL.load_config(config_path)

            with using_forge_operations(device=memory_management.cpu, dtype=memory_management.vae_dtype()):
                model = IntegratedAutoencoderKL.from_config(config)

            load_state_dict(model, state_dict, ignore_start='loss.')
            return model
        if component_name.startswith('text_encoder') and cls_name in ['CLIPTextModel', 'CLIPTextModelWithProjection']:
            assert isinstance(state_dict, dict) and len(state_dict) > 16, 'You do not have CLIP state dict!'

            from transformers import CLIPTextConfig, CLIPTextModel
            config = CLIPTextConfig.from_pretrained(config_path)

            to_args = dict(device=memory_management.cpu, dtype=memory_management.text_encoder_dtype())

            with modeling_utils.no_init_weights():
                with using_forge_operations(**to_args, manual_cast_enabled=True):
                    model = IntegratedCLIP(CLIPTextModel, config, add_text_projection=True).to(**to_args)

            load_state_dict(model, state_dict, ignore_errors=[
                'transformer.text_projection.weight',
                'transformer.text_model.embeddings.position_ids',
                'logit_scale'
            ], log_name=cls_name)

            return model

        if component_name.startswith('text_encoder') and cls_name in ['ChatGLMModel']:
            assert isinstance(state_dict, dict) and len(state_dict) > 16, 'You do not have CLIP state dict!'

            config_clip = ChatGLMConfig.from_pretrained(config_path)
            to_args = dict(device=memory_management.cpu, dtype=memory_management.text_encoder_dtype())

            with modeling_utils.no_init_weights():
                with modeling_utils.no_init_weights():
                    model = IntegratedCLIPKolors(ChatGLMModel, config=config_clip, add_text_projection=True).to(**to_args)

            load_state_dict(model, state_dict, ignore_errors=[
                'transformer.text_projection.weight',
                'transformer.text_model.embeddings.position_ids',
                'logit_scale'
            ], log_name=cls_name)

            return model

        if cls_name == 'T5EncoderModel':
            assert isinstance(state_dict, dict) and len(state_dict) > 16, 'You do not have T5 state dict!'

            from backend.nn.t5 import IntegratedT5
            config = read_arbitrary_config(config_path)

            storage_dtype = memory_management.text_encoder_dtype()
            state_dict_dtype = memory_management.state_dict_dtype(state_dict)

            if state_dict_dtype in [torch.float8_e4m3fn, torch.float8_e5m2, 'nf4', 'fp4', 'gguf']:
                printF(name=MasterName.get_master_name(), info="Using Detected T5 Data Type: {}".format(state_dict_dtype)).printf()
                storage_dtype = state_dict_dtype
                if state_dict_dtype in ['nf4', 'fp4', 'gguf']:
                    printF(name=MasterName.get_master_name(), info="Using pre-quant state dict!").printf()
                    if state_dict_dtype in ['gguf']:
                        beautiful_print_gguf_state_dict_statics(state_dict)
            else:
                printF(name=MasterName.get_master_name(), info="Using Default T5 Data Type: {}".format(storage_dtype)).printf()

            if storage_dtype in ['nf4', 'fp4', 'gguf']:
                with modeling_utils.no_init_weights():
                    with using_forge_operations(device=memory_management.cpu, dtype=memory_management.text_encoder_dtype(), manual_cast_enabled=False, bnb_dtype=storage_dtype):
                        model = IntegratedT5(config)
            else:
                with modeling_utils.no_init_weights():
                    with using_forge_operations(device=memory_management.cpu, dtype=storage_dtype, manual_cast_enabled=True):
                        model = IntegratedT5(config)

            load_state_dict(model, state_dict, log_name=cls_name, ignore_errors=['transformer.encoder.embed_tokens.weight', 'logit_scale'])

            return model
        if cls_name in ['UNet2DConditionModel', 'FluxTransformer2DModel']:
            assert isinstance(state_dict, dict) and len(state_dict) > 16, 'You do not have model state dict!'

            model_loader = None
            if cls_name == 'UNet2DConditionModel':
                if "kolors" in guess.huggingface_repo.lower():
                    model_loader = lambda c: IntegratedKolors2DConditionModel.from_config(c)
                else:
                    model_loader = lambda c: IntegratedUNet2DConditionModel.from_config(c)
            if cls_name == 'FluxTransformer2DModel':
                from backend.nn.flux import IntegratedFluxTransformer2DModel
                model_loader = lambda c: IntegratedFluxTransformer2DModel(**c)

            unet_config = guess.unet_config.copy()
            state_dict_parameters = memory_management.state_dict_parameters(state_dict)
            state_dict_dtype = memory_management.state_dict_dtype(state_dict)

            storage_dtype = memory_management.unet_dtype(model_params=state_dict_parameters, supported_dtypes=guess.supported_inference_dtypes)

            unet_storage_dtype_overwrite = backend.args.dynamic_args.get('forge_unet_storage_dtype')

            if unet_storage_dtype_overwrite is not None:
                storage_dtype = unet_storage_dtype_overwrite
            elif state_dict_dtype in [torch.float8_e4m3fn, torch.float8_e5m2, 'nf4', 'fp4', 'gguf']:
                printF(name=MasterName.get_master_name(), info="Using Detected UNet Type: {}".format(state_dict_dtype)).printf()
                storage_dtype = state_dict_dtype
                if state_dict_dtype in ['nf4', 'fp4', 'gguf']:
                    printF(name=MasterName.get_master_name(), info="Using pre-quant state dict!").printf()
                    if state_dict_dtype in ['gguf']:
                        beautiful_print_gguf_state_dict_statics(state_dict)

            load_device = memory_management.get_torch_device()
            computation_dtype = memory_management.get_computation_dtype(load_device, parameters=state_dict_parameters, supported_dtypes=guess.supported_inference_dtypes)
            offload_device = memory_management.unet_offload_device()

            if storage_dtype in ['nf4', 'fp4', 'gguf']:
                initial_device = memory_management.unet_inital_load_device(parameters=state_dict_parameters, dtype=computation_dtype)
                with using_forge_operations(device=initial_device, dtype=computation_dtype, manual_cast_enabled=False, bnb_dtype=storage_dtype):
                    model = model_loader(unet_config)
            else:
                initial_device = memory_management.unet_inital_load_device(parameters=state_dict_parameters, dtype=storage_dtype)
                need_manual_cast = storage_dtype != computation_dtype
                to_args = dict(device=initial_device, dtype=storage_dtype)

                with using_forge_operations(**to_args, manual_cast_enabled=need_manual_cast):
                    model = model_loader(unet_config).to(**to_args)

            load_state_dict(model, state_dict)

            if hasattr(model, '_internal_dict'):
                model._internal_dict = unet_config
            else:
                model.config = unet_config

            model.storage_dtype = storage_dtype
            model.computation_dtype = computation_dtype
            model.load_device = load_device
            model.initial_device = initial_device
            model.offload_device = offload_device

            return model

    printF(name=MasterName.get_master_name(), info="Skipped: {} = {}.{}".format(component_name, lib_name, cls_name)).printf()
    return None


def replace_state_dict(sd, asd, guess):
    vae_key_prefix = guess.vae_key_prefix[0]
    text_encoder_key_prefix = guess.text_encoder_key_prefix[0]

    if 'enc.blk.0.attn_k.weight' in asd:
        wierd_t5_format_from_city96 = {
            "enc.": "encoder.",
            ".blk.": ".block.",
            "token_embd": "shared",
            "output_norm": "final_layer_norm",
            "attn_q": "layer.0.SelfAttention.q",
            "attn_k": "layer.0.SelfAttention.k",
            "attn_v": "layer.0.SelfAttention.v",
            "attn_o": "layer.0.SelfAttention.o",
            "attn_norm": "layer.0.layer_norm",
            "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
            "ffn_up": "layer.1.DenseReluDense.wi_1",
            "ffn_down": "layer.1.DenseReluDense.wo",
            "ffn_gate": "layer.1.DenseReluDense.wi_0",
            "ffn_norm": "layer.1.layer_norm",
        }
        wierd_t5_pre_quant_keys_from_city96 = ['shared.weight']
        asd_new = {}
        for k, v in asd.items():
            for s, d in wierd_t5_format_from_city96.items():
                k = k.replace(s, d)
            asd_new[k] = v
        for k in wierd_t5_pre_quant_keys_from_city96:
            asd_new[k] = asd_new[k].dequantize_as_pytorch_parameter()
        asd.clear()
        asd = asd_new

    if "decoder.conv_in.weight" in asd:
        keys_to_delete = [k for k in sd if k.startswith(vae_key_prefix)]
        for k in keys_to_delete:
            del sd[k]
        for k, v in asd.items():
            sd[vae_key_prefix + k] = v

    if 'text_model.encoder.layers.0.layer_norm1.weight' in asd:
        keys_to_delete = [k for k in sd if k.startswith(f"{text_encoder_key_prefix}clip_l.")]
        for k in keys_to_delete:
            del sd[k]
        for k, v in asd.items():
            sd[f"{text_encoder_key_prefix}clip_l.transformer.{k}"] = v

    if 'encoder.block.0.layer.0.SelfAttention.k.weight' in asd:
        keys_to_delete = [k for k in sd if k.startswith(f"{text_encoder_key_prefix}t5xxl.")]
        for k in keys_to_delete:
            del sd[k]
        for k, v in asd.items():
            sd[f"{text_encoder_key_prefix}t5xxl.transformer.{k}"] = v

    if 'encoder.layers.0.input_layernorm.weight' in asd:
        keys_to_delete = [k for k in sd if k.startswith(f"{text_encoder_key_prefix}")]
        for k in keys_to_delete:
            del sd[k]
        for k, v in asd.items():
            sd[f"{text_encoder_key_prefix}clip_l.transformer.{k}"] = v

    return sd


def preprocess_state_dict(sd):
    if any("double_block" in k for k in sd.keys()):
        if not any(k.startswith("model.diffusion_model") for k in sd.keys()):
            sd = {f"model.diffusion_model.{k}": v for k, v in sd.items()}

    return sd


def split_state_dict(sd, additional_state_dicts: list = None, skip=False):
    if not skip:
        sd = load_torch_file(sd)
    sd = preprocess_state_dict(sd)
    guess = huggingface_guess.guess(sd)

    if isinstance(additional_state_dicts, list):
        for asd in additional_state_dicts:
            asd = load_torch_file(asd)
            sd = replace_state_dict(sd, asd, guess)

    guess.clip_target = guess.clip_target(sd)

    state_dict = {
        guess.unet_target: try_filter_state_dict(sd, guess.unet_key_prefix),
        guess.vae_target: try_filter_state_dict(sd, guess.vae_key_prefix)
    }

    sd = guess.process_clip_state_dict(sd)

    for k, v in guess.clip_target.items():
        state_dict[v] = try_filter_state_dict(sd, [k + '.'])

    state_dict['ignore'] = sd

    print_dict = {k: len(v) for k, v in state_dict.items()}
    printF(name=MasterName.get_master_name(), info="StateDict Keys: {}".format(print_dict)).printf()

    del state_dict['ignore']

    return state_dict, guess


@torch.inference_mode()
def forge_loader(sd, additional_state_dicts=None, skip=False):
    try:
        state_dicts, estimated_config = split_state_dict(sd, additional_state_dicts=additional_state_dicts,  skip=skip)
    except:
        raise ValueError('Failed to recognize model type!')

    repo_name = estimated_config.huggingface_repo
    # printF(name=MasterName.get_master_name(),
    #        info="state_dicts: {} | estimated_config: {} | repo_name: {}".format(state_dicts, "", repo_name)).printf()

    local_path = os.path.join(dir_path, 'huggingface', repo_name)
    config: dict = DiffusionPipeline.load_config(local_path)
    printF(name=MasterName.get_master_name(),
           info="config: {}".format(config)).printf()

    huggingface_components = {}
    for component_name, v in config.items():
        if isinstance(v, list) and len(v) == 2:
            lib_name, cls_name = v
            component_sd = state_dicts.get(component_name, None)
            # printF(name=MasterName.get_master_name(),
            #        info="estimated_config: {} | component_name: {} | lib_name: {} | cls_name: {} | local_path: {} | component_sd: {}".format(estimated_config.__dict__, component_name, lib_name, cls_name, local_path, component_sd)).printf()
            component = load_huggingface_component(estimated_config, component_name, lib_name, cls_name, local_path, component_sd)
            if component_sd is not None:
                del state_dicts[component_name]
            if component is not None:
                huggingface_components[component_name] = component

    for M in possible_models:
        if any(isinstance(estimated_config, x) for x in M.matched_guesses):
            return M(estimated_config=estimated_config, huggingface_components=huggingface_components)

    printF(name=MasterName.get_master_name(), info="Failed to recognize model type!").printf()
    return None
