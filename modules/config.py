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

        if deprecated_config_dict.get("default_model", None) == 'juggernautXL_v9Rundiffusionphoto2.safetensors':
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
path_upscale_models = get_dir_or_set_default('path_upscale_models', '../models/upscale_models/')
path_inpaint = get_dir_or_set_default('path_inpaint', '../models/inpaint/')
path_controlnet = get_dir_or_set_default('path_controlnet', '../models/controlnet/')
path_clip_vision = get_dir_or_set_default('path_clip_vision', '../models/clip_vision/')
path_fooocus_expansion = get_dir_or_set_default('path_fooocus_expansion',
                                                '../models/prompt_expansion/fooocus_expansion')
path_wildcards = get_dir_or_set_default('path_wildcards', '../wildcards/')
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
default_controlnet_canny_name = 'control-lora-canny-rank128.safetensors'
default_controlnet_depth_name = 'control-lora-depth-rank128.safetensors'

preset_filenames = []
model_filenames = []
lora_filenames = []
controlnet_lora_canny_filenames = []
controlnet_lora_depth_filenames = []

default_model_preset_name = "default"

default_base_model_name = default_model = get_config_item_or_set_default(
    key='default_model',
    default_value='juggernautXL_v9Rundiffusionphoto2.safetensors',
    validator=lambda x: isinstance(x, str)
)

previous_default_models = get_config_item_or_set_default(
    key='previous_default_models',
    default_value=[],
    validator=lambda x: isinstance(x, list) and all(isinstance(k, str) for k in x)
)
default_refiner_model_name = default_refiner = get_config_item_or_set_default(
    key='default_refiner',
    default_value='None',
    validator=lambda x: isinstance(x, str)
)
default_refiner_switch = get_config_item_or_set_default(
    key='default_refiner_switch',
    default_value=0.8,
    validator=lambda x: isinstance(x, numbers.Number) and 0 <= x <= 1
)
default_loras_min_weight = get_config_item_or_set_default(
    key='default_loras_min_weight',
    default_value=-2,
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
        '512*512', '768*768',
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
    "default_refiner": "refiner_model",
    "default_refiner_switch": "refiner_switch",
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
    return f'{a}×{b} <span style="color: grey;"> \U00002223 {a // g}:{b // g}</span>'


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
sdxl_model_filenames = []
sd15_model_filenames = []
lora_filenames = []
sdxl_lora_filenames = []
sd15_lora_filenames = []
sdxl_lcm_lora = 'sdxl_lcm_lora.safetensors'
sdxl_lightning_lora = 'sdxl_lightning_4step_lora.safetensors'
loras_metadata_remove = [sdxl_lcm_lora, sdxl_lightning_lora]
wildcard_filenames = []


def get_model_filenames(folder_paths, name_filter=None):
    extensions = ['.pth', '.ckpt', '.bin', '.safetensors', '.fooocus.patch']
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
    global preset_filenames, model_filenames, lora_filenames, controlnet_lora_canny_filenames, controlnet_lora_depth_filenames, sd15_model_filenames, sdxl_model_filenames, sdxl_lora_filenames, sd15_lora_filenames
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
        sdxl_model_filenames += get_model_filenames(modelfile_path, name_filter=j)
        sdxl_lora_filenames += get_model_filenames(lorafile_path, name_filter=j)

    sd15_model_filenames = sorted(set(sd15_model_filenames), key=sd15_model_filenames.index)
    sd15_lora_filenames = sorted(set(sd15_lora_filenames), key=sd15_lora_filenames.index)

    sdxl_model_filenames = sorted(set(sdxl_model_filenames), key=sdxl_model_filenames.index)
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
        file_name='sdxl_lightning_4step_lora.safetensors'
    )
    return 'sdxl_lightning_4step_lora.safetensors'


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
