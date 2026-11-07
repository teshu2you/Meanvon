import os
import json
import math
import numbers
import sys
import tempfile
import adapter.args_manager
from modules.flags import MetadataScheme, Performance
import modules.sdxl_styles
from modules.util import get_files_from_folder, get_config_or_set_default, get_config_item_or_set_default, config_dict, \
    makedirs_with_log
from modules.flags import OutputFormat, Performance, MetadataScheme
from modules.model_loader import load_file_from_url
from modules.util import get_files_from_folder
from util.printf import printF, MasterName


def get_config_path(key, default_value):
    env = os.getenv(key)
    if env is not None and isinstance(env, str):
        printF(name=MasterName.get_master_name(), info="Environment: {} = {}".format(key, env)).printf()
        return env
    else:
        return os.path.abspath(default_value)


config_path = get_config_path('config_path', "./config.txt")
printF(name=MasterName.get_master_name(), info="[Info] config_path = {}".format(config_path)).printf()

config_example_path = get_config_path('config_example_path', "config_modification_tutorial.txt")
printF(name=MasterName.get_master_name(), info="[Info] config_example_path = {}".format(config_example_path)).printf()

config_dict = {}
always_save_keys = []
visited_keys = []

config_path_mre = "paths.json"


def load_paths(paths_filename):
    paths_dict = {
        'modelfile_path': '../models/checkpoints/',
        'lorafile_path': '../models/loras/',
        'embeddings_path': '../models/embeddings/',
        'clip_vision_path': '../models/clip_vision/',
        'controlnet_path': '../models/controlnet/',
        'vae_approx_path': '../models/vae_approx/',
        'fooocus_expansion_path': '../models/prompt_expansion/fooocus_expansion/',
        'upscale_models_path': '../models/upscale_models/',
        'inpaint_models_path': '../models/inpaint/',
        'styles_path': '../sdxl_styles/',
        'wildcards_path': '../wildcards/',
        'temp_outputs_path': '../outputs/'
    }

    if os.path.exists(paths_filename):
        with open(paths_filename, encoding='utf-8') as paths_file:
            try:
                paths_obj = json.load(paths_file)
                if 'paths_checkpoints' in paths_obj:
                    paths_dict['modelfile_path'] = paths_obj['paths_checkpoints']
                if 'paths_loras' in paths_obj:
                    paths_dict['lorafile_path'] = paths_obj['paths_loras']
                if 'path_embeddings' in paths_obj:
                    paths_dict['embeddings_path'] = paths_obj['path_embeddings']
                if 'path_clip_vision' in paths_obj:
                    paths_dict['clip_vision_path'] = paths_obj['path_clip_vision']
                if 'path_controlnet' in paths_obj:
                    paths_dict['controlnet_path'] = paths_obj['path_controlnet']
                if 'path_vae_approx' in paths_obj:
                    paths_dict['vae_approx_path'] = paths_obj['path_vae_approx']
                if 'path_vae' in paths_obj:
                    paths_dict['vae_path'] = paths_obj['path_vae']
                if 'path_text_encoder' in paths_obj:
                    paths_dict['text_encoder_path'] = paths_obj['path_text_encoder']
                if 'path_fooocus_expansion' in paths_obj:
                    paths_dict['fooocus_expansion_path'] = paths_obj['path_fooocus_expansion']
                if 'path_upscale_models' in paths_obj:
                    paths_dict['upscale_models_path'] = paths_obj['path_upscale_models']
                if 'path_inpaint_models' in paths_obj:
                    paths_dict['inpaint_models_path'] = paths_obj['path_inpaint_models']
                if 'path_styles' in paths_obj:
                    paths_dict['styles_path'] = paths_obj['path_styles']
                if 'path_wildcards' in paths_obj:
                    paths_dict['wildcards_path'] = paths_obj['path_wildcards']
                if 'path_outputs' in paths_obj:
                    paths_dict['temp_outputs_path'] = paths_obj['path_outputs']

            except Exception as e:
                printF(name=MasterName.get_master_name(), info="[ERROR] load_paths, e: {}".format(e)).printf()
            finally:
                paths_file.close()

    return paths_dict


try:
    with open(os.path.abspath(f'./presets/default.json'), "r", encoding="utf-8") as json_file:
        config_dict.update(json.load(json_file))
except Exception as e:
    printF(name=MasterName.get_master_name(), info="[ERROR] Load default preset failed. e: {}".format(e)).printf()

try:
    if os.path.exists(config_path_mre):
        with open(config_path_mre, "r", encoding="utf-8") as json_file:
            config_dict = load_paths(config_path_mre)
    elif os.path.exists(config_path):
        with open(os.path.abspath(f'./presets/default.json'), "r", encoding="utf-8") as json_file:
            config_dict.update(json.load(json_file))
except Exception as e:
    printF(name=MasterName.get_master_name(), info="[ERROR] Load default preset failed. e: {}".format(e)).printf()

try:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as json_file:
            config_dict.update(json.load(json_file))
            always_save_keys = list(config_dict.keys())
except Exception as e:
    printF(name=MasterName.get_master_name(),
           info="[ERROR] Failed to load config file {}. The reason is: {}".format(config_path, e)).printf()
    printF(name=MasterName.get_master_name(), info="[ERROR] Please make sure that:").printf()
    printF(name=MasterName.get_master_name(),
           info="[ERROR] 1. The file {} is a valid text file, and you have access to read it.".format(
               config_path)).printf()
    printF(name=MasterName.get_master_name(),
           info="[ERROR] 2. Use \'\\\\\' instead of \'\\' when describing paths.").printf()
    printF(name=MasterName.get_master_name(), info="[ERROR] 3. There is no ',' before the last '}'").printf()
    printF(name=MasterName.get_master_name(), info="[ERROR] 4. All key/value formats are correct.").printf()


def try_load_deprecated_user_path_config():
    global config_dict

    if not os.path.exists('user_path_config.txt'):
        return

    try:
        deprecated_config_dict = json.load(open('user_path_config.txt', "r", encoding="utf-8"))

        def replace_config(old_key, new_key):
            if old_key in deprecated_config_dict:
                config_dict[new_key] = deprecated_config_dict[old_key]
                del deprecated_config_dict[old_key]

        replace_config('modelfile_path', 'paths_checkpoints')
        replace_config('lorafile_path', 'paths_loras')
        replace_config('embeddings_path', 'path_embeddings')
        replace_config('vae_approx_path', 'path_vae_approx')
        replace_config('upscale_models_path', 'path_upscale_models')
        replace_config('inpaint_models_path', 'path_inpaint')
        replace_config('controlnet_models_path', 'path_controlnet')
        replace_config('clip_vision_models_path', 'path_clip_vision')
        replace_config('fooocus_expansion_path', 'path_fooocus_expansion')
        replace_config('temp_outputs_path', 'path_outputs')

        if deprecated_config_dict.get("default_model", None) == 'SDXL_Juggernaut-XI-byRunDiffusion.safetensors':
            os.replace('user_path_config.txt', 'user_path_config-deprecated.txt')
            print('Config updated successfully in silence. '
                  'A backup of previous config is written to "user_path_config-deprecated.txt".')
            return

        if input("Newer models and configs are available. "
                 "Download and update files? [Y/n]:") in ['n', 'N', 'No', 'no', 'NO']:
            config_dict.update(deprecated_config_dict)
            print('Loading using deprecated old models and deprecated old configs.')
            return
        else:
            os.replace('user_path_config.txt', 'user_path_config-deprecated.txt')
            print('Config updated successfully by user. '
                  'A backup of previous config is written to "user_path_config-deprecated.txt".')
            return
    except Exception as e:
        print('Processing deprecated config failed')
        print(e)
    return


try_load_deprecated_user_path_config()


def get_presets():
    preset_folder = 'presets'
    presets = ['initial']
    if not os.path.exists(preset_folder):
        print('No presets found.')
        return presets

    return presets + [f[:f.index('.json')] for f in os.listdir(preset_folder) if f.endswith('.json')]


def try_get_preset_content(preset):
    if isinstance(preset, str):
        preset_path = os.path.abspath(f'./presets/{preset}.json')
        try:
            if os.path.exists(preset_path):
                with open(preset_path, "r", encoding="utf-8") as json_file:
                    json_content = json.load(json_file)
                    print(f'Loaded preset: {preset_path}')
                    return json_content
            else:
                raise FileNotFoundError
        except Exception as e:
            print(f'Load preset [{preset_path}] failed')
            print(e)
    return {}


available_presets = get_presets()
preset = adapter.args_manager.args.preset
config_dict.update(try_get_preset_content(preset))


def get_config_from_model_preset(preset_name):
    model_preset_dict = {}
    model_preset_file = os.path.abspath(f'./presets/{preset_name}.json')
    try:
        if os.path.exists(model_preset_file):
            with open(model_preset_file, "r", encoding="utf-8") as f:
                model_preset_dict = json.load(f)
    except Exception as e:
        printF(name=MasterName.get_master_name(), info="[Error] Load preset config failed : {}".format(e)).printf()
    return model_preset_dict


def get_path_output(obj="images") -> str:
    """
    Checking output path argument and overriding default path.
    """
    global config_dict
    path_output = get_dir_or_set_default('path_outputs_' + obj, '../outputs/' + obj + "/")
    printF(name=MasterName.get_master_name(), info="[Parameters] path_output = {}".format(path_output)).printf()
    if adapter.args_manager.args.output_path:
        printF(name=MasterName.get_master_name(), info="[CONFIG] Overriding config value path_outputs with {}".format(
            adapter.args_manager.args.output_path)).printf()
        config_dict['path_outputs'] = path_output = adapter.args_manager.args.output_path
    return path_output


def get_dir_or_set_default(key, default_value, as_array=False, make_directory=False):
    global config_dict, visited_keys, always_save_keys

    if key not in visited_keys:
        visited_keys.append(key)

    if key not in always_save_keys:
        always_save_keys.append(key)

    v = os.getenv(key)
    if v is not None:
        printF(name=MasterName.get_master_name(), info="[Environment] {} = {}".format(key, v)).printf()
        config_dict[key] = v
    else:
        v = config_dict.get(key, None)

    if isinstance(v, str):
        if make_directory:
            makedirs_with_log(v)
        if os.path.exists(v) and os.path.isdir(v):
            return v if not as_array else [v]
    elif isinstance(v, list):
        if make_directory:
            for d in v:
                makedirs_with_log(d)
        if all([os.path.exists(d) and os.path.isdir(d) for d in v]):
            return v

    if v is not None:
        print(
            f'Failed to load config key: {json.dumps({key: v})} is invalid or does not exist; will use {json.dumps({key: default_value})} instead.')
    if isinstance(default_value, list):
        dp = []
        for path in default_value:
            abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
            dp.append(abs_path)
            os.makedirs(abs_path, exist_ok=True)
    else:
        dp = os.path.abspath(os.path.join(os.path.dirname(__file__), default_value))
        os.makedirs(dp, exist_ok=True)
        if as_array:
            dp = [dp]
    config_dict[key] = dp
    return dp


paths_checkpoints = get_dir_or_set_default('paths_checkpoints', ['../models/checkpoints/'], True)
paths_loras = get_dir_or_set_default('paths_loras', ['../models/loras/'], True)
path_embeddings = get_dir_or_set_default('path_embeddings', '../models/embeddings/')
path_vae_approx = get_dir_or_set_default('path_vae_approx', '../models/vae_approx/')
path_vae = get_dir_or_set_default('path_vae', '../models/vae/')
path_llm = get_dir_or_set_default('path_llm', '../models/LLM/')
path_text_encoder = get_dir_or_set_default('path_text_encoder', '../models/text_encoder/')
path_upscale_models = get_dir_or_set_default('path_upscale_models', '../models/upscale_models/')
path_inpaint = get_dir_or_set_default('path_inpaint', '../models/inpaint/')
path_controlnet = get_dir_or_set_default('path_controlnet', '../models/controlnet/')
path_clip_vision = get_dir_or_set_default('path_clip_vision', '../models/clip_vision/')
path_fooocus_expansion = get_dir_or_set_default('path_fooocus_expansion',
                                                '../models/prompt_expansion/fooocus_expansion')
path_wildcards = get_dir_or_set_default('path_wildcards', '../wildcards/')
path_safety_checker = get_dir_or_set_default('path_safety_checker', '../models/safety_checker/')
path_outputs = get_path_output()

modelfile_path = get_dir_or_set_default('modelfile_path', '../models/checkpoints/', as_array=True)
lorafile_path = get_dir_or_set_default('lorafile_path', '../models/loras/', as_array=True)
embeddings_path = get_dir_or_set_default('embeddings_path', '../models/embeddings/')
vae_approx_path = get_dir_or_set_default('vae_approx_path', '../models/vae_approx/')
upscale_models_path = get_dir_or_set_default('upscale_models_path', '../models/upscale_models/')
inpaint_models_path = get_dir_or_set_default('inpaint_models_path', '../models/inpaint/')

clip_vision_models_path = get_dir_or_set_default('clip_vision_models_path', '../models/clip_vision/')
fooocus_expansion_path = get_dir_or_set_default('fooocus_expansion_path',
                                                '../models/prompt_expansion/fooocus_expansion')
temp_outputs_path = get_dir_or_set_default('temp_outputs_path', '../outputs/images/')
clip_vision_path = get_config_or_set_default('clip_vision_path', '../models/clip_vision/')
styles_path = get_config_or_set_default('styles_path', '../sdxl_styles/')
last_prompt_path = os.path.join(path_outputs, 'last_prompt.json')

controlnet_path = get_config_or_set_default('controlnet_path', '../models/controlnet/')
controlnet_models_path = get_dir_or_set_default('controlnet_models_path', '../models/controlnet/', as_array=True)
controlnet_lora_path = get_dir_or_set_default('controlnet_lora_path', '../models/controlnet/', as_array=True)

model_presets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../presets/'))
path_style_samples = get_dir_or_set_default('path_style_samples', '../sdxl_styles_samples/')
# 生成视频的输出文件夹
# Output Folder for Generated Videos
# vid_output_folder = 'content/outputs'
path_images_outputs = get_path_output(obj="images")
path_videos_outputs = get_path_output(obj="videos")

path_sadtalker_checkpoint = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../extensions/sadtalker/checkpoints/'))
path_sadtalker_config = os.path.abspath(os.path.join(os.path.dirname(__file__), '../extensions/sadtalker/src/config/'))

path_videos_sadtalker_outputs = os.path.join(path_videos_outputs, 'sadtalker')

with open(config_path, "w", encoding="utf-8") as json_file:
    json.dump(config_dict, json_file, indent=4)


def get_config_item_or_set_default(key, default_value, validator, disable_empty_as_none=False):
    global config_dict, visited_keys

    if key not in visited_keys:
        visited_keys.append(key)

    v = os.getenv(key)
    if v is not None:
        print(f"Environment: {key} = {v}")
        config_dict[key] = v

    if key not in config_dict:
        config_dict[key] = default_value
        return default_value

    v = config_dict.get(key, None)
    if not disable_empty_as_none:
        if v is None or v == '':
            v = 'None'
    if validator(v):
        return v
    else:
        if v is not None:
            print(
                f'Failed to load config key: {json.dumps({key: v})} is invalid; will use {json.dumps({key: default_value})} instead.')
        config_dict[key] = default_value
        return default_value


def init_temp_path(path: str | None, default_path: str) -> str:
    if adapter.args_manager.args.temp_path:
        path = adapter.args_manager.args.temp_path

    if path != '' and path != default_path:
        try:
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            os.makedirs(path, exist_ok=True)
            printF(name=MasterName.get_master_name(), info="[Parameters] Using temp path = {}".format(path)).printf()
            return path
        except Exception as e:
            printF(name=MasterName.get_master_name(),
                   info="[ERROR] Could not create temp path {}. Reason: {}".format(path, e)).printf()
            printF(name=MasterName.get_master_name(),
                   info="[ERROR] Using default temp path {} instead.".format(default_path)).printf()

    os.makedirs(default_path, exist_ok=True)
    return default_path


default_temp_path = os.path.join(tempfile.gettempdir(), 'MeanVon')
temp_path = init_temp_path(get_config_item_or_set_default(
    key='temp_path',
    default_value=default_temp_path,
    validator=lambda x: isinstance(x, str),
), default_temp_path)
temp_path_cleanup_on_launch = get_config_item_or_set_default(
    key='temp_path_cleanup_on_launch',
    default_value=True,
    validator=lambda x: isinstance(x, bool)
)

os.makedirs(temp_outputs_path, exist_ok=True)

default_clip_vision_name = 'clip_vision_g.safetensors'

default_flux1_vae_name = 'ae.safetensors'
default_flux1_text_encoder_clip = 'clip_l.safetensors'
default_flux1_text_encoder_t5xxl = 't5xxl_fp16.safetensors'

default_flux2_vae_name = 'flux2-vae.safetensors'
default_flux2_text_encoder_mistral = None   # dev 32B 用
default_flux2_text_encoder_qwen3_4b = 'qwen_3_4b.safetensors'  # klein 4B 用
default_flux2_text_encoder_qwen3_8b = 'qwen_3_8b_fp8mixed.safetensors'  # klein 9B 用（你现在需要的）

default_zimage_vae_name = 'zimage_vae.safetensors'
default_zimage_text_encoder_clip = 'qwen_3_4b.safetensors'

default_qwenimage_vae_name = 'qwen_image_vae.safetensors'
default_qwenimage_text_encoder = 'qwen_2.5_vl_7b.safetensors'

# HunyuanDiT 的辅助权重（unet-only checkpoint 必须）
# default_hunyuandit_vae_name = "sdxl-vae-fp16-fix.safetensors"
# default_hunyuandit_text_encoder_clip = "clip_l.safetensors"
# default_hunyuandit_text_encoder_mt5 = "mt5xxl_fp16.safetensors"

# Krea  qwen_image_vae
default_krea2_vae_name = "wan_2.1_vae.safetensors"
default_krea2_text_encoder = "qwen_3vl_4b_bf16.safetensors"

default_mossimage_vae_name = "moss-image-vae.safetensors"
default_mossimage_text_encoder = "qwen3.5_9b_int8_convrot.safetensors"

default_anima_vae_name = "qwen_image_vae.safetensors"
default_anima_text_encoder = "qwen_3_06b_base.safetensors"

# FLUX.3（f3-image-small）：与 FLUX.2 共用 128ch packed VAE；TE 是 Qwen3-VL 4B
default_flux3_vae_name = "flux2-vae.safetensors"
default_flux3_sd_text_encoder = "qwen_3vl_4b_int8_convrot.safetensors"
default_flux3_base_text_encoder = "qwen_3vl_4b_bf16.safetensors"

# ===== Anima 的参考图（结构参考）行为 =====
# Anima 的底座是 Cosmos-Predict2（MiniTrainDIT），上游没有“参考图”通道，
# 本项目的做法是把参考 latent 当成额外一帧拼进时间轴。
# 实测结论（Anima_Aesthetic-v1.1 + 结构图）：
#   参考帧路径能让结构几乎完全正确，但模型会把“相邻的干净帧”当成未来帧去插值，
#   于是细节被抹平 → 出图明显发糊/重影。
# 因此默认关闭参考帧，图生图回退到**通用 img2img 初始 latent**（结构靠 denoise 控制）：
#   default_anima_use_reference
#       False -> 标准 img2img 初始 latent（默认，出图清晰；强度看 img2img_denoise，
#                0.55~0.70 更贴结构，0.80~0.95 更锐利/更自由）
#       True  -> 使用参考帧（结构最准，但细节偏软）
#   default_anima_ref_frame_order
#       "prepend" -> [参考帧, 生成帧]，取最后一帧（推荐，比旧行为清晰）
#       "append"  -> [生成帧, 参考帧]，取第一帧（旧行为，最容易发糊/重影）
default_anima_use_reference = False
default_anima_ref_frame_order = "prepend"

default_hunyuandit_vae_name = None
default_hunyuandit_text_encoder_clip = None
default_hunyuandit_text_encoder_mt5 = None

default_controlnet_canny_name = 'control-lora-canny-rank128.safetensors'
default_controlnet_depth_name = 'control-lora-depth-rank128.safetensors'

default_kolors_chatglm_name = "chatglm3-4bit.safetensors"

preset_filenames = []
model_filenames = []
lora_filenames = []
controlnet_lora_canny_filenames = []
controlnet_lora_depth_filenames = []

default_model_preset_name = "default"

model_types = [
    # "Stable_Zero123",
    # "SD15_instructpix2pix",
    # "SD15",
    # "SD20",
    # "SD21UnclipL",
    # "SD21UnclipH",
    # "SDXL_instructpix2pix",
    # "SDXLRefiner",
    "SDXL",
    # "SSD1B",
    # "KOALA_700M",
    # "KOALA_1B",
    # "Segmind_Vega",
    # "SD_X4Upscaler",
    # "Stable_Cascade_C",
    # "Stable_Cascade_B",
    # "SV3D_u",
    # "SV3D_p",
    # "SD3",
    # "StableAudio",
    # "AuraFlow",
    # "QwenImage3",
    # "QwenImage2_1",
    # "HunyuanDiT",
    "Krea2",
    "Flux1_Schnell",
    "Flux1_DEV",
    "Flux2_DEV",
    "Flux2_Klein",
    "Flux3_Klein_Step_Distilled",
    "Flux3_Klein_Base",
    # "Kolors",
    "ZImage_Turbo",
    "ZImage_Base",
    # "ErnieImage",
    # ---- Moss Image 分支拆分：Base（基础版）/ SFT（微调版） ----
    "MossImage_Base",
    "MossImage_SFT",
    # ---- Anima ----
    "Anima",
]

# ===== 图生图（img2img）denoise 的按模型族配置 =====
# 不同模型族的“噪声语义”完全不同，同一个 denoise 数值效果差别巨大：
#   * flow matching（Flux / ZImage / Anima / Moss / Krea2 / QwenImage …）
#       起始 sigma ≈ denoise，初始 latent（结构图）权重 = 1 - denoise
#       → denoise=1.00 权重 0%（等于纯文生图，看起来“图生图没生效”）
#       → denoise=0.94 权重 6%（几乎看不出结构）
#       → denoise=0.60~0.70 才是“像原图但更好看”
#   * SD / SDXL（VP 噪声，A1111 语义）
#       0.30~0.70 常见，0.94 同样几乎等于重画
#   * 蒸馏模型（Flux-Schnell / ZImage-Turbo / Moss-SFT / 少步数模型）
#       去噪预算少，denoise 过高会崩/糊 → 0.35~0.55
# key 用小写子串匹配（model_type_selector 优先，其次 get_model_file_type 的返回值）；
# 长 key 优先匹配（flux2_klein 先于 flux2，zimage_turbo 先于 zimage）。
img2img_denoise_profiles = {
    "sdxl":            dict(recommend=0.55, minimum=0.25, maximum=0.80, note="SDXL：A1111 语义，>0.8 基本重画"),
    "sd15":            dict(recommend=0.55, minimum=0.25, maximum=0.80, note="SD1.5：A1111 语义"),
    "kolors":          dict(recommend=0.55, minimum=0.25, maximum=0.80, note="Kolors：SD 系语义"),
    "flux1_schnell":   dict(recommend=0.45, minimum=0.25, maximum=0.60, note="Flux-Schnell：蒸馏 4 步，denoise 过高会崩"),
    "flux1_dev":       dict(recommend=0.60, minimum=0.35, maximum=0.72, note="Flux-dev：flow，sigma≈denoise"),
    "flux2_klein":     dict(recommend=0.50, minimum=0.30, maximum=0.65, note="Flux2-Klein：蒸馏，denoise 过高会崩"),
    "flux2":           dict(recommend=0.60, minimum=0.35, maximum=0.72, note="Flux2：flow，sigma≈denoise"),
    "flux3_klein_base": dict(recommend=0.60, minimum=0.35, maximum=0.72, note="Flux3-Klein Base：flow，sigma≈denoise"),
    "flux3_klein":     dict(recommend=0.50, minimum=0.30, maximum=0.65, note="Flux3-Klein：蒸馏，denoise 过高会崩"),
    "flux3":           dict(recommend=0.55, minimum=0.30, maximum=0.70, note="Flux3：flow，sigma≈denoise"),
    "zimage_turbo":    dict(recommend=0.45, minimum=0.25, maximum=0.60, note="ZImage-Turbo：蒸馏 8 步，denoise 过高会糊"),
    "zimage":          dict(recommend=0.60, minimum=0.35, maximum=0.72, note="ZImage-Base：flow，sigma≈denoise"),
    "anima":           dict(recommend=0.65, minimum=0.40, maximum=0.72, note="Anima：flow，sigma≈denoise，>0.72 结构基本丢失"),
    "mossimage_sft":   dict(recommend=0.50, minimum=0.30, maximum=0.65, note="MOSS-Image SFT：denoise 过高会崩"),
    "mossimage":       dict(recommend=0.55, minimum=0.30, maximum=0.70, note="MOSS-Image Base：flow"),
    "krea2":           dict(recommend=0.60, minimum=0.35, maximum=0.72, note="Krea2：flow，sigma≈denoise"),
    "qwenimage":       dict(recommend=0.60, minimum=0.35, maximum=0.72, note="Qwen-Image：flow，sigma≈denoise"),
    "hunyuan":         dict(recommend=0.55, minimum=0.30, maximum=0.70, note="HunyuanDiT：denoise 过高会糊"),
}

# 未匹配到模型族时的保守默认（相当于 SDXL 语义）
img2img_denoise_default_profile = dict(recommend=0.60, minimum=0.25, maximum=0.80,
                                       note="通用默认：flow/SD 都能用的中间值")


# ===== 模型类型 → (sampler, scheduler) 预设 =====
# 与 webui_server.MODEL_TYPE_PRESETS 的 sampler_pairs 对齐，但放在 config 里，
# 让后端（procedure/worker）也能在 switch_sampler=True 时按模型族强制套用——
# 不依赖 Gradio 事件是否把下拉框改成功。
# 解析顺序：长 key 优先（flux3_klein_base 先于 flux3），取第一个“两项都合法”的组合。
model_type_sampler_pairs = {
    "flux3_klein_base": [("euler", "simple"), ("dpmpp_2m", "normal")],
    "flux3":            [("euler", "simple"), ("dpmpp_2m", "normal")],
    "flux2_klein":      [("euler", "simple")],
    "flux2":            [("euler", "simple"), ("euler", "beta")],
    "flux1_schnell":    [("euler", "simple")],
    "flux1_dev":        [("euler", "beta"), ("euler", "karras")],
    "flux1":            [("euler", "simple")],
    "zimage_turbo":     [("euler_ancestral", "beta"), ("dpmpp_sde", "beta")],
    "zimage":           [("dpmpp_2m", "karras"), ("euler", "simple")],
    "anima":            [("res_multistep", "beta"), ("er_sde", "beta"), ("euler", "simple")],
    "mossimage_sft":    [("euler", "sgm_uniform")],
    "mossimage":        [("euler", "sgm_uniform"), ("er_sde", "simple")],
    "krea2":            [("euler", "simple")],
    "qwenimage":        [("euler", "simple")],
    "hunyuan":          [("dpmpp_2m", "karras")],
    "sdxl":             [("dpmpp_2m", "karras"), ("dpmpp_sde", "karras")],
}

model_type_sampler_pairs_default = [("euler", "simple")]


def get_model_type_sampler_pairs(model_type_name):
    """返回该模型族的 (sampler, scheduler) 候选列表（长 key 优先匹配）。"""
    key = _norm_model_key(model_type_name)
    if key:
        for raw in sorted(model_type_sampler_pairs, key=len, reverse=True):
            if _norm_model_key(raw) in key:
                return list(model_type_sampler_pairs[raw])
    return list(model_type_sampler_pairs_default)


def _norm_model_key(text):
    """'FLUX.1 [dev]' -> 'flux1dev'；'ZImage-Turbo' -> 'zimageturbo'。"""
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


def get_img2img_denoise_profile(*names):
    """按模型族返回 img2img denoise 的推荐值/可用区间。

    参数按优先级给出候选名字（如 UI 的 model_type_selector、preset 标签、
    以及 get_model_file_type() 的返回值），第一个命中的生效。

    名字会去掉标点后再做子串匹配（'FLUX.1 [dev]' / 'flux1_dev' / 'Flux1_DEV'
    都能命中 'flux1dev'），长 key 优先，避免 flux2_klein 被 flux2 抢走。

    返回 dict(recommend, minimum, maximum, note, matched)。
    """
    keys = sorted(img2img_denoise_profiles.keys(), key=len, reverse=True)
    norm_keys = [(k, _norm_model_key(k)) for k in keys]

    for name in names:
        key = _norm_model_key(name)
        if not key:
            continue
        for raw, norm in norm_keys:
            if norm and norm in key:
                prof = dict(img2img_denoise_default_profile)
                prof.update(img2img_denoise_profiles[raw])
                prof["matched"] = raw
                return prof

    prof = dict(img2img_denoise_default_profile)
    prof["matched"] = "default"
    return prof

# ---- ZImage 分支拆分：Turbo（蒸馏快速版）/ Base（基础版） ----
# 公共默认（两分支共用，可在各自 dict 里覆盖）
_zimage_common = {
    'vae_name': default_zimage_vae_name,
    'text_encoder': default_zimage_text_encoder_clip,
}

# Turbo 分支配置
zimage_turbo_config = {
    **_zimage_common,
    'model_type': 'ZImage_Turbo',
    'default_steps': 8,
    'default_cfg_scale': 1.0,
}

# Base 分支配置
zimage_base_config = {
    **_zimage_common,
    'model_type': 'ZImage_Base',
    'default_steps': 30,
    'default_cfg_scale': 5.0,
}

zimage_branch_configs = {
    'ZImage_Turbo': zimage_turbo_config,
    'ZImage_Base': zimage_base_config,
}


def is_zimage(model_type: str) -> bool:
    """是否为任意 ZImage 分支。"""
    return model_type in zimage_branch_configs


def is_zimage_turbo(model_type: str) -> bool:
    return model_type == 'ZImage_Turbo'


def is_zimage_base(model_type: str) -> bool:
    return model_type == 'ZImage_Base'

def get_zimage_config(model_type: str) -> dict:
    """按分支类型返回对应配置，未命中返回空 dict。"""
    return zimage_branch_configs.get(model_type, {})


# ---- Moss Image 分支（Base / SFT）----
# 公共默认（两分支共用，可在各自 dict 里覆盖）
_mossimage_common = {
    'vae_name': default_mossimage_vae_name,
    'text_encoder': default_mossimage_text_encoder,
}

# Base 分支配置
mossimage_base_config = {
    **_mossimage_common,
    'model_type': 'MossImage_Base',
    'default_steps': 30,
    'default_cfg_scale': 5.0,
}

# SFT 分支配置
mossimage_sft_config = {
    **_mossimage_common,
    'model_type': 'MossImage_SFT',
    'default_steps': 30,
    'default_cfg_scale': 5.0,
}

mossimage_branch_configs = {
    'MossImage_Base': mossimage_base_config,
    'MossImage_SFT': mossimage_sft_config,
}


def is_mossimage(model_type: str) -> bool:
    """是否为任意 Moss Image 分支。"""
    return model_type in mossimage_branch_configs


def is_mossimage_base(model_type: str) -> bool:
    return model_type == 'MossImage_Base'


def is_mossimage_sft(model_type: str) -> bool:
    return model_type == 'MossImage_SFT'


def get_mossimage_config(model_type: str) -> dict:
    """按分支类型返回对应配置，未命中返回空 dict。"""
    return mossimage_branch_configs.get(model_type, {})


# ---- Anima ----
anima_config = {
    'model_type': 'Anima',
    'vae_name': default_anima_vae_name,
    'text_encoder': default_anima_text_encoder,
    'default_steps': 30,
    'default_cfg_scale': 5.0,
}


def is_anima(model_type: str) -> bool:
    return model_type == 'Anima'


def get_anima_config(model_type: str) -> dict:
    return anima_config if model_type == 'Anima' else {}


default_model_type = "SDXL"

default_base_model_name = default_model = get_config_item_or_set_default(
    key='default_model',
    default_value='SDXL_Juggernaut-XI-byRunDiffusion.safetensors',
    validator=lambda x: isinstance(x, str)
)

previous_default_models = get_config_item_or_set_default(
    key='previous_default_models',
    default_value=[],
    validator=lambda x: isinstance(x, list) and all(isinstance(k, str) for k in x)
)

default_loras_min_weight = get_config_item_or_set_default(
    key='default_loras_min_weight',
    default_value=0.2,
    validator=lambda x: isinstance(x, numbers.Number) and -10 <= x <= 10
)
default_loras_max_weight = get_config_item_or_set_default(
    key='default_loras_max_weight',
    default_value=2,
    validator=lambda x: isinstance(x, numbers.Number) and -10 <= x <= 10
)
default_loras = get_config_item_or_set_default(
    key='default_loras',
    default_value=[
        [
            True,
            "None",
            1.0
        ],
        [
            True,
            "None",
            1.0
        ],
        [
            True,
            "None",
            1.0
        ],
        [
            True,
            "None",
            1.0
        ],
        [
            True,
            "None",
            1.0
        ]
    ],
    validator=lambda x: isinstance(x, list) and all(
        len(y) == 3 and isinstance(y[0], bool) and isinstance(y[1], str) and isinstance(y[2], numbers.Number)
        or len(y) == 2 and isinstance(y[0], str) and isinstance(y[1], numbers.Number)
        for y in x)
)
default_loras = [(y[0], y[1], y[2]) if len(y) == 3 else (True, y[0], y[1]) for y in default_loras]
default_max_lora_number = get_config_item_or_set_default(
    key='default_max_lora_number',
    default_value=len(default_loras) if isinstance(default_loras, list) and len(default_loras) > 0 else 5,
    validator=lambda x: isinstance(x, int) and x >= 1
)
default_cfg_scale = get_config_item_or_set_default(
    key='default_cfg_scale',
    default_value=7.0,
    validator=lambda x: isinstance(x, numbers.Number)
)
default_sample_sharpness = get_config_item_or_set_default(
    key='default_sample_sharpness',
    default_value=2.0,
    validator=lambda x: isinstance(x, numbers.Number)
)
default_sampler = get_config_item_or_set_default(
    key='default_sampler',
    default_value='dpmpp_2m_sde_gpu',
    validator=lambda x: x in modules.flags.sampler_list
)
default_scheduler = get_config_item_or_set_default(
    key='default_scheduler',
    default_value='karras',
    validator=lambda x: x in modules.flags.scheduler_list
)
default_styles = get_config_item_or_set_default(
    key='default_styles',
    default_value=[
        "Fooocus V2",
        "Fooocus Enhance",
        "Fooocus Sharp"
    ],
    validator=lambda x: isinstance(x, list) and all(y in modules.sdxl_styles.legal_style_names for y in x)
)
default_prompt_negative = get_config_item_or_set_default(
    key='default_prompt_negative',
    default_value='',
    validator=lambda x: isinstance(x, str),
    disable_empty_as_none=True
)
default_prompt = get_config_item_or_set_default(
    key='default_prompt',
    default_value='',
    validator=lambda x: isinstance(x, str),
    disable_empty_as_none=True
)
default_performance = get_config_item_or_set_default(
    key='default_performance',
    default_value=Performance.SPEED.value,
    validator=lambda x: x in Performance.list()
)
default_advanced_checkbox = get_config_item_or_set_default(
    key='default_advanced_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool)
)
default_max_image_number = get_config_item_or_set_default(
    key='default_max_image_number',
    default_value=32,
    validator=lambda x: isinstance(x, int) and x >= 1
)
default_output_format = get_config_item_or_set_default(
    key='default_output_format',
    default_value='png',
    validator=lambda x: x in OutputFormat.list()
)
default_image_factory_advanced_checkbox = get_config_item_or_set_default(
    key='default_image_factory_advanced_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool)
)

default_video_factory_advanced_checkbox = get_config_item_or_set_default(
    key='default_video_factory_advanced_checkbox',
    default_value=False,
    validator=lambda x: isinstance(x, bool)
)
default_image_number = get_config_item_or_set_default(
    key='default_image_number',
    default_value=1,
    validator=lambda x: isinstance(x, int) and 1 <= x <= default_max_image_number
)
checkpoint_downloads = get_config_item_or_set_default(
    key='checkpoint_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items())
)
lora_downloads = get_config_item_or_set_default(
    key='lora_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items())
)
embeddings_downloads = get_config_item_or_set_default(
    key='embeddings_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items())
)
available_aspect_ratios = get_config_item_or_set_default(
    key='available_aspect_ratios',
    default_value=[
        '768*768',
        '704*1408', '704*1344', '768*1344', '768*1280', '832*1216', '832*1152',
        '896*1152', '896*1088', '960*1088', '960*1024', '1024*1024', '1024*960',
        '1088*960', '1088*896', '1152*896', '1152*832', '1216*832', '1280*768',
        '1344*768', '1344*704', '1408*704', '1472*704', '1536*640', '1600*640',
        '1664*576', '1728*576'
    ],
    validator=lambda x: isinstance(x, list) and all('*' in v for v in x) and len(x) > 1
)
default_aspect_ratio = get_config_item_or_set_default(
    key='default_aspect_ratio',
    default_value='1024*1024' if '1024*1024' in available_aspect_ratios else available_aspect_ratios[0],
    validator=lambda x: x in available_aspect_ratios
)
default_inpaint_engine_version = get_config_item_or_set_default(
    key='default_inpaint_engine_version',
    default_value='v2.6',
    validator=lambda x: x in modules.flags.inpaint_engine_versions
)
default_cfg_tsnr = get_config_item_or_set_default(
    key='default_cfg_tsnr',
    default_value=7.0,
    validator=lambda x: isinstance(x, numbers.Number)
)
default_overwrite_step = get_config_item_or_set_default(
    key='default_overwrite_step',
    default_value=-1,
    validator=lambda x: isinstance(x, int)
)
default_overwrite_switch = get_config_item_or_set_default(
    key='default_overwrite_switch',
    default_value=-1,
    validator=lambda x: isinstance(x, int)
)
example_inpaint_prompts = get_config_item_or_set_default(
    key='example_inpaint_prompts',
    default_value=[
        'highly detailed face', 'detailed girl face', 'detailed man face', 'detailed hand', 'beautiful eyes'
    ],
    validator=lambda x: isinstance(x, list) and all(isinstance(v, str) for v in x)
)
default_save_metadata_to_images = get_config_item_or_set_default(
    key='default_save_metadata_to_images',
    default_value=True,
    validator=lambda x: isinstance(x, bool)
)
default_metadata_scheme = get_config_item_or_set_default(
    key='default_metadata_scheme',
    default_value=MetadataScheme.FOOOCUS.value,
    validator=lambda x: x in [y[1] for y in modules.flags.metadata_scheme if y[1] == x]
)
metadata_created_by = get_config_item_or_set_default(
    key='metadata_created_by',
    default_value='',
    validator=lambda x: isinstance(x, str)
)


example_inpaint_prompts = [[x] for x in example_inpaint_prompts]

config_dict["default_loras"] = default_loras = default_loras[:default_max_lora_number] + [['None', 1.0] for _ in range(
    default_max_lora_number - len(default_loras))]

# mapping config to meta parameter
possible_preset_keys = {
    "default_model": "base_model",
    "previous_default_models": "previous_default_models",
    "default_loras_min_weight": "default_loras_min_weight",
    "default_loras_max_weight": "default_loras_max_weight",
    "default_loras": "<processed>",
    "default_cfg_scale": "guidance_scale",
    "default_sample_sharpness": "sharpness",
    "default_sampler": "sampler",
    "default_scheduler": "scheduler",
    "default_overwrite_step": "steps",
    "default_performance": "performance",
    "default_image_number": "image_number",
    "default_prompt": "prompt",
    "default_prompt_negative": "negative_prompt",
    "default_styles": "styles",
    "default_aspect_ratio": "resolution",
    "default_save_metadata_to_images": "default_save_metadata_to_images",
    "checkpoint_downloads": "checkpoint_downloads",
    "embeddings_downloads": "embeddings_downloads",
    "lora_downloads": "lora_downloads"
}

REWRITE_PRESET = False

if REWRITE_PRESET and isinstance(adapter.args_manager.args.preset, str):
    save_path = 'presets/' + adapter.args_manager.args.preset + '.json'
    with open(save_path, "w", encoding="utf-8") as json_file:
        json.dump({k: config_dict[k] for k in possible_preset_keys}, json_file, indent=4)
    printF(name=MasterName.get_master_name(), info="Preset saved to {}. Exiting ...".format(save_path)).printf()
    exit(0)


def add_ratio(x):
    a, b = x.replace('*', ' ').split(' ')[:2]
    a, b = int(a), int(b)
    g = math.gcd(a, b)
    return f'{a}×{b} \U00002223 {a // g}:{b // g}'


default_aspect_ratio = add_ratio(default_aspect_ratio)
available_aspect_ratios = [add_ratio(x) for x in available_aspect_ratios]

# Only write config in the first launch.
if not os.path.exists(config_path):
    with open(config_path, "w", encoding="utf-8") as json_file:
        json.dump({k: config_dict[k] for k in always_save_keys}, json_file, indent=4)

# Always write tutorials.
if preset is None:
    # Do not overwrite user config if preset is applied.
    with open(config_path, "w", encoding="utf-8") as json_file:
        json.dump({k: config_dict[k] for k in visited_keys}, json_file, indent=4)

with open(config_example_path, "w", encoding="utf-8") as json_file:
    cpa = config_path.replace("\\", "\\\\")
    json_file.write(f'You can modify your "{cpa}" using the below keys, formats, and examples.\n'
                    f'Do not modify this file. Modifications in this file will not take effect.\n'
                    f'This file is a tutorial and example. Please edit "{cpa}" to really change any settings.\n'
                    + 'Remember to split the paths with "\\\\" rather than "\\", '
                      'and there is no "," before the last "}". \n\n\n')
    json.dump({k: config_dict[k] for k in visited_keys}, json_file, indent=4)

os.makedirs(path_outputs, exist_ok=True)

model_filenames = []
sd_model_filenames = []
sd15_model_filenames = []
lora_filenames = []
sdxl_lora_filenames = []
sd15_lora_filenames = []
sdxl_lcm_lora = 'sdxl_lcm_lora.safetensors'
sdxl_lightning_lora = 'sdxl_lightning_4step_lora.safetensors'
sdxl_hyper_sd_lora = 'Hyper-SDXL-4steps-lora.safetensors'
loras_metadata_remove = [sdxl_lcm_lora, sdxl_lightning_lora, sdxl_hyper_sd_lora]
wildcard_filenames = []


def get_model_filenames(folder_paths, name_filter=None):
    extensions = ['.pth', '.ckpt', '.bin', '.gguf', '.safetensors', '.fooocus.patch']
    files = []
    for folder in folder_paths:
        files += get_files_from_folder(folder, extensions, name_filter)
    return files


def get_preset_filenames(model_presets_path, name_filter=None):
    preset_filenames = ['default']
    all_model_presets_files = get_files_from_folder(model_presets_path, ['.json'], name_filter)
    if all_model_presets_files:
        for model_preset in all_model_presets_files:
            _n = model_preset.split(".")[0]
            if _n == "default":
                continue
            preset_filenames.append(_n)
    return preset_filenames


def update_all_model_names():
    global preset_filenames, model_filenames, lora_filenames, controlnet_lora_canny_filenames, controlnet_lora_depth_filenames, sd15_model_filenames, sd_model_filenames, sdxl_lora_filenames, sd15_lora_filenames
    global wildcard_filenames, available_presets
    model_filenames = get_model_filenames(paths_checkpoints)
    lora_filenames = get_model_filenames(paths_loras)
    wildcard_filenames = get_files_from_folder(path_wildcards, ['.txt'])
    available_presets = get_presets()
    preset_filenames = get_preset_filenames(model_presets_path)

    sd15_list_str = ["SD15", "sd15"]
    sdxl_list_str = ["SDXL", "sd_xl", "XL"]

    for i in sd15_list_str:
        sd15_model_filenames += get_model_filenames(modelfile_path, name_filter=i)
        sd15_lora_filenames += get_model_filenames(lorafile_path, name_filter=i)

    for j in sdxl_list_str:
        sd_model_filenames += get_model_filenames(modelfile_path, name_filter=j)
        sdxl_lora_filenames += get_model_filenames(lorafile_path, name_filter=j)

    sd15_model_filenames = sorted(set(sd15_model_filenames), key=sd15_model_filenames.index)
    sd15_lora_filenames = sorted(set(sd15_lora_filenames), key=sd15_lora_filenames.index)

    sd_model_filenames = sorted(set(sd_model_filenames), key=sd_model_filenames.index)
    sdxl_lora_filenames = sorted(set(sdxl_lora_filenames), key=sdxl_lora_filenames.index)

    controlnet_lora_canny_filenames = get_model_filenames(controlnet_lora_path, name_filter="canny")
    controlnet_lora_depth_filenames = get_model_filenames(controlnet_lora_path, name_filter="depth")

    return


def downloading_inpaint_models(v):
    assert v in modules.flags.inpaint_engine_versions

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth',
        model_dir=path_inpaint,
        file_name='fooocus_inpaint_head.pth'
    )
    head_file = os.path.join(path_inpaint, 'fooocus_inpaint_head.pth')
    patch_file = None

    if v == 'v1':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint.fooocus.patch')

    if v == 'v2.5':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint_v25.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint_v25.fooocus.patch')

    if v == 'v2.6':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint_v26.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint_v26.fooocus.patch')

    return head_file, patch_file


def downloading_sdxl_lcm_lora():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors',
        model_dir=paths_loras[0],
        file_name=sdxl_lcm_lora
    )
    return sdxl_lcm_lora


def downloading_sdxl_turbo_lora():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/SDXL_turbo_lora_v1.safetensors',
        model_dir=paths_loras[0],
        file_name='SDXL_turbo_lora_v1.safetensors'
    )
    return 'SDXL_turbo_lora_v1.safetensors'


def downloading_sdxl_lightning_lora():
    load_file_from_url(
        url='https://hf-mirror.com/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_lora.safetensors?download=true',
        model_dir=paths_loras[0],
        file_name=modules.flags.PerformanceLoRA.LIGHTNING.value
    )
    return modules.flags.PerformanceLoRA.LIGHTNING.value

def downloading_sdxl_hyper_sd_lora():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sdxl_hyper_sd_4step_lora.safetensors',
        model_dir=paths_loras[0],
        file_name=modules.flags.PerformanceLoRA.HYPER_SD.value
    )
    return modules.flags.PerformanceLoRA.HYPER_SD.value

def downloading_controlnet_canny():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors',
        model_dir=path_controlnet,
        file_name='control-lora-canny-rank128.safetensors'
    )
    return os.path.join(path_controlnet, 'control-lora-canny-rank128.safetensors')


def downloading_controlnet_cpds():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors',
        model_dir=path_controlnet,
        file_name='fooocus_xl_cpds_128.safetensors'
    )
    return os.path.join(path_controlnet, 'fooocus_xl_cpds_128.safetensors')


def downloading_ip_adapters(v=""):
    assert v in ['ip', 'face']

    results = []

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors',
        model_dir=path_clip_vision,
        file_name='clip_vision_vit_h.safetensors'
    )
    results += [os.path.join(path_clip_vision, 'clip_vision_vit_h.safetensors')]

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors',
        model_dir=path_controlnet,
        file_name='fooocus_ip_negative.safetensors'
    )
    results += [os.path.join(path_controlnet, 'fooocus_ip_negative.safetensors')]

    if v == 'ip':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin',
            model_dir=path_controlnet,
            file_name='ip-adapter-plus_sdxl_vit-h.bin'
        )
        results += [os.path.join(path_controlnet, 'ip-adapter-plus_sdxl_vit-h.bin')]

    if v == 'face':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin',
            model_dir=path_controlnet,
            file_name='ip-adapter-plus-face_sdxl_vit-h.bin'
        )
        results += [os.path.join(path_controlnet, 'ip-adapter-plus-face_sdxl_vit-h.bin')]

    return results


def downloading_upscale_model():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin',
        model_dir=path_upscale_models,
        file_name='fooocus_upscaler_s409985e5.bin'
    )
    return os.path.join(path_upscale_models, 'fooocus_upscaler_s409985e5.bin')


def downloading_safety_checker_model():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/stable-diffusion-safety-checker.bin',
        model_dir=path_safety_checker,
        file_name='stable-diffusion-safety-checker.bin'
    )
    return os.path.join(path_safety_checker, 'stable-diffusion-safety-checker.bin')

svd_config = {
    # svd_xt  svd
    "version": "svd",
    "svd": {
        "num_frames": 1,
        "num_steps": 1,
        "model_config": "repositories/generative-models/scripts/sampling/configs/svd.yaml"
    },
    "svd_xt": {
        "num_frames": 25,
        "num_steps": 30,
        "model_config": "repositories/generative-models/scripts/sampling/configs/svd_xt.yaml"
    },
}

update_all_model_names()
