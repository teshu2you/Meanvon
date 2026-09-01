import ssl
from util.printf import printF, MasterName
from config.webuiConfig import *
from util.printf import MasterName, printF
import random
ssl._create_default_https_context = ssl._create_unverified_context
import ast
import json
import copy
import socket
import warnings
import shared
import modules.config
from version import main_version
import modules.html
# import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.style_sorter as style_sorter
import modules.meta_parser
import adapter.args_manager
from os.path import exists
from collections.abc import Mapping
import json
from fastapi import FastAPI
from modules.settings import default_settings, infer_args
from modules.resolutions import get_resolution_new_string
from modules.sdxl_styles import legal_style_names, style_keys, fooocus_expansion, \
    hot_style_keys, normalize_key, migrate_style_from_v1, default_legal_style_names
from modules.private_logger import get_current_html_path
from modules.util import get_current_log_path, get_previous_log_path, is_json
from modules.ui_gradio_extensions import reload_javascript
from modules.auth import auth_enabled, check_auth
import ast
from resources import *
import socket
from procedure.worker_ui_patch import task_manager
from adapter.task_queue import QueueTask, TaskQueue
from config import *

worker_queue: TaskQueue = None
queue_task: QueueTask = None
last_model_name = None

warnings.filterwarnings('ignore')
GALLERY_ID_INPUT = 0
GALLERY_ID_REVISION = 1
GALLERY_ID_OUTPUT = 2

# 检查 viewer_* 函数是否存在，如果不存在则使用默认行为
switch_js = """
(x) => {
    try {
        if(x){
            if(typeof viewer_to_bottom === 'function') {
                viewer_to_bottom(100);
                viewer_to_bottom(500);
            } else {
                // 默认滚动到底部
                setTimeout(() => window.scrollTo(0, document.body.scrollHeight), 100);
            }
        } else {
            if(typeof viewer_to_top === 'function') {
                viewer_to_top();
            } else {
                // 默认滚动到顶部
                window.scrollTo(0, 0);
            }
        }
    } catch(e) {
        console.warn('viewer function error:', e);
        // 降级处理
        if(x) {
            window.scrollTo(0, document.body.scrollHeight);
        } else {
            window.scrollTo(0, 0);
        }
    }
    return x;
}
"""

down_js = """
() => {
    try {
        if(typeof viewer_to_bottom === 'function') {
            viewer_to_bottom();
        } else {
            // 默认滚动到底部
            setTimeout(() => window.scrollTo(0, document.body.scrollHeight), 100);
        }
    } catch(e) {
        console.warn('viewer_to_bottom error:', e);
        window.scrollTo(0, document.body.scrollHeight);
    }
}
"""

def local_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0)
    try:
        sock.connect(("10.0.0.1", 1))
        host_ip = sock.getsockname()[0]
    except Exception as e:
        host_ip = "127.0.0.1"
    finally:
        sock.close()
    return host_ip


tmp_bug = "./.tmp"
os.makedirs(tmp_bug, exist_ok=True)

blankfile_common = "./.tmp/blank.txt"
with open(blankfile_common, 'w') as savefile:
    savefile.write("")

ini_dir = "./.ini"
os.makedirs(ini_dir, exist_ok=True)

log_dir = "./.logs"
os.makedirs(log_dir, exist_ok=True)
logfile_bug = f"{log_dir}/output.log"
sys.stdout = Logger(logfile_bug)


def get_task(*args):
    args = list(args)
    args.pop(0)
    print(">>> after pop, args count:", len(args))
    return task_manager.AsyncTask(args=args)


def generate_clicked(task: task_manager.AsyncTask):
    import ldm_patched.modules.model_management as model_management

    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False

    # outputs=[progress_html, progress_window, progress_gallery, gallery]

    if len(task.args) == 0:
        return

    execution_start_time = time.perf_counter()
    finished = False
    # task = worker.AsyncTask(args=list(args))
    # task = task_manager.AsyncTask(args=list(args))

    yield gr.update(visible=True, value=modules.html.make_progress_html(1, 'Waiting for task to start ...')), \
        gr.update(visible=True, value=None), \
        gr.update(visible=True, value="Ready to process..."), \
        gr.update(visible=False), \
        gr.update(), \
        gr.update(), \
        gr.update(), \
        gr.update(value=None), \
        gr.update()

    task_manager.async_tasks.append(task)

    while not finished:
        time.sleep(0.01)
        # print(f"task.yields --------- 111: {task.yields}")
        if len(task.yields) > 0:
            # print(f"task.yields ---------- 222: {task.yields}")
            flag, product = task.yields.pop(0)

            if flag == "preview":
                # help bad internet connection by skipping duplicated preview
                if len(task.yields) > 0:  # if we have the next item
                    if (
                        task.yields[0][0] == "preview"
                    ):  # if the next item is also a preview
                        continue
                percentage, title, image, img_pp, img_rr = product
                yield (
                    gr.update(
                        visible=True,
                        value=modules.html.make_progress_html(percentage, title),
                    ),
                    gr.update(visible=True, value=image)
                    if image is not None
                    else gr.update(),
                    gr.update(
                        visible=True,
                        value="No."
                        + str(img_pp)
                        + " processing...           |           "
                        + str(img_rr)
                        + "  image(s) pending!",
                    ),
                    gr.update(visible=False),
                    gr.update(),
                    gr.update(),
                    gr.update(open=True),
                    gr.update(),
                    gr.update(),
                )

            elif flag == "metadatas":  # 改为 elif
                yield (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(value=product),
                    gr.update(selected=GALLERY_ID_OUTPUT),
                )

            elif flag == "results":  # 改为 elif
                yield (
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True, value="Partially done"),
                    gr.update(visible=False),
                    gr.update(),
                    gr.update(visible=True, value=product)
                    if product is not None
                    else gr.update(visible=False),
                    gr.update(open=True),
                    gr.update(),
                    gr.update(selected=GALLERY_ID_OUTPUT),
                )

            elif flag == "finish":  # 改为 elif，并添加 continue
                yield (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True, value="All done"),
                    gr.update(visible=True),
                    gr.update(value=product),
                    gr.update(value=product),
                    gr.update(open=False),
                    gr.update(),
                    gr.update(selected=GALLERY_ID_OUTPUT),
                )
                finished = True

                # delete Fooocus temp images, only keep gradio temp images
                if adapter.args_manager.args.disable_image_log:
                    for filepath in product:
                        if isinstance(filepath, str) and os.path.exists(filepath):
                            os.remove(filepath)
                continue  # 添加 continue，避免后续代码继续执行

    execution_time = time.perf_counter() - execution_start_time
    print(f"Total time: {execution_time:.2f} seconds")
    return


def metadata_to_ctrls(metadata, ctrls):
    # important webui parameters!
    if not isinstance(metadata, Mapping):
        return ctrls

    def _to_float(value, default=None):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _to_int(value, default=None):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def _to_bool(value, default=False):
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value) if value is not None else default

    def _clamp(value, low, high, default=None):
        value = _to_float(value, default)
        if value is None:
            return default
        return max(low, min(high, value))

    def _clamp_int(value, low, high, default=None):
        value = _to_int(value, default)
        if value is None:
            return default
        return max(low, min(high, value))

    def _to_list(value, default=None):
        if value is None:
            return default
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except Exception:
                try:
                    return json.loads(value)
                except Exception:
                    return default
        return default

    if 'prompt' in metadata:
        ctrls[2] = metadata.get('prompt')
    if 'negative_prompt' in metadata:
        ctrls[3] = metadata.get('negative_prompt')
    if 'styles' in metadata:
        ctrls[4] = _to_list(metadata.get('styles'), ctrls[4])
    elif 'style' in metadata:
        ctrls[4] = migrate_style_from_v1(metadata.get('style'))
    if 'performance' in metadata:
        ctrls[5] = metadata.get('performance')
    if 'width' in metadata and 'height' in metadata:
        ctrls[6] = get_resolution_new_string(metadata.get('width'), metadata.get('height'))
    elif 'resolution' in metadata:
        ctrls[6] = metadata.get('resolution')
    if 'image_number' in metadata:
        ctrls[7] = _clamp_int(metadata.get('image_number'),
                              1, modules.config.default_max_image_number, ctrls[7])
    if 'seed' in metadata:
        ctrls[8] = metadata.get('seed')
    if 'sharpness' in metadata:
        ctrls[9] = _clamp(metadata.get('sharpness'), 0, 30, ctrls[9])
    # ctrls[10] switch_sampler skip
    if 'sampler_name' in metadata:
        ctrls[11] = metadata.get('sampler_name')
    elif 'sampler' in metadata:
        ctrls[11] = metadata.get('sampler')
    if 'scheduler' in metadata:
        ctrls[12] = metadata.get('scheduler')
    if 'steps' in metadata:
        ctrls[13] = _clamp_int(metadata.get('steps'), 1, 200, ctrls[13])
        ctrls[14] = ctrls[13]
    if 'switch' in metadata:
        _steps = _to_float(metadata.get('steps'), 1.0) or 1.0
        _switch = _to_float(metadata.get('switch'), ctrls[15])
        if _switch is not None:
            ctrls[15] = _clamp(round(_switch / _steps, 2), 0.2, 1.0, ctrls[15])
    if 'cfg' in metadata:
        ctrls[16] = _clamp(metadata.get('cfg'), 1, 30, ctrls[16])
    if 'guidance_scale' in metadata:
        ctrls[16] = _clamp(metadata.get('guidance_scale'), 1, 30, ctrls[16])

    if 'base_model' in metadata:
        _tmp = metadata.get('base_model')
        if ".safetensors" not in _tmp and ".gguf" not in _tmp and _tmp not in ['None', 'none', 'Not Exist!->']:
            ctrls[17] = _tmp + ".safetensors"
        else:
            ctrls[17] = _tmp
    elif 'base_model_name' in metadata:
        _tmp = metadata.get('base_model_name')
        if ".safetensors" not in _tmp and ".gguf" not in _tmp and _tmp not in ['None', 'none', 'Not Exist!->']:
            ctrls[17] = _tmp + ".safetensors"
        else:
            ctrls[17] = _tmp
    if 'refiner_model' in metadata:
        _tmp = metadata.get('refiner_model')
        if ".safetensors" not in _tmp and ".gguf" not in _tmp and _tmp not in ['None', 'none', 'Not Exist!->']:
            ctrls[18] = _tmp + ".safetensors"
        else:
            ctrls[18] = _tmp
    elif 'refiner_model_name' in metadata:
        _tmp = metadata.get('refiner_model_name')
        if ".safetensors" not in _tmp and ".gguf" not in _tmp and _tmp not in ['None', 'none', 'Not Exist!->']:
            ctrls[18] = _tmp + ".safetensors"
        else:
            ctrls[18] = _tmp
    if 'base_clip_skip' in metadata:
        ctrls[19] = _clamp_int(metadata.get('base_clip_skip'), -10, -1, ctrls[19])
    if 'refiner_clip_skip' in metadata:
        ctrls[20] = _clamp_int(metadata.get('refiner_clip_skip'), -10, -1, ctrls[20])
    if 'refiner_switch' in metadata:
        ctrls[21] = _clamp(metadata.get('refiner_switch'), 0.1, 1.0, ctrls[21])

    lora_begin_idx = 22
    lora_num = 5
    for lrn in range(lora_num):
        index = 'lora_combined_' + str(lrn + 1)
        if index in metadata:
            ctrls[lora_begin_idx] = True
            kv = metadata.get(index).split(":")
            _tmp = kv[0].strip()
            if ".safetensors" not in _tmp:
                ctrls[lora_begin_idx + 1] = _tmp + ".safetensors"
            else:
                ctrls[lora_begin_idx + 1] = _tmp
            if len(kv) > 1:
                ctrls[lora_begin_idx + 2] = _clamp(
                    kv[1].strip(),
                    modules.config.default_loras_min_weight,
                    modules.config.default_loras_max_weight,
                    1.0
                )
            else:
                ctrls[lora_begin_idx + 2] = 1
        else:
            ctrls[lora_begin_idx] = False
            ctrls[lora_begin_idx + 1] = "None"
            ctrls[lora_begin_idx + 2] = 1
        lora_begin_idx += 3

    if 'model_type_selector' in metadata and len(ctrls) > 122:
        ctrls[122] = metadata.get('model_type_selector')
    if 'seed_random' in metadata and len(ctrls) > 123:
        ctrls[123] = _to_bool(metadata.get('seed_random'), ctrls[123])

    printF(name=MasterName.get_master_name(),
           info="[Parameters] AFTER--> ctrls: {} - {}".format(len(ctrls), ctrls)).printf()
    return ctrls


def load_prompt_handler(_file, *args):
    ctrls = list(args)
    printF(name=MasterName.get_master_name(),
           info="[Parameters] BEFORE--> ctrls: {} - {}".format(len(ctrls), ctrls)).printf()
    path = _file.name
    if path.endswith('.json'):
        with open(path, encoding='utf-8') as json_file:
            try:
                json_obj = json.load(json_file)
                printF(name=MasterName.get_master_name(), info="[Parameters] json_obj = {}".format(json_obj)).printf()
                ctrls = metadata_to_ctrls(json_obj, ctrls)
            except Exception as e:
                print(f'json -- load_prompt_handler, e: {e} ctrls: {len(ctrls)} - {ctrls}')
            finally:
                json_file.close()
    else:
        with open(path, 'rb') as image_file:
            image = Image.open(image_file)
            image_file.close()

            if path.endswith('.png') and 'parameters' in image.info:
                metadata_string = image.info['parameters']
            elif path.endswith('.jpg') and 'parameters' in image.info:
                metadata_bytes = image.info['parameters']
                metadata_string = metadata_bytes.decode('utf-8').split('\0')[0]
            else:
                metadata_string = None

            if metadata_string is not None:
                try:
                    # print(f'metadata_string:{metadata_string}')
                    metadata = json.loads(metadata_string)
                    printF(name=MasterName.get_master_name(),
                           info="[Parameters] metadata = {}".format(metadata)).printf()
                    if metadata.get("loras"):
                        for idx, mmm in enumerate(metadata["loras"]):
                            metadata["lora_combined_" + str(idx + 1)] = mmm[0] + ":" + str(mmm[1])
                    ctrls = metadata_to_ctrls(metadata, ctrls)
                except Exception as e:
                    printF(name=MasterName.get_master_name(),
                           info="[ERROR] load_prompt_handler e = {} -  {} - {}".format(e, len(ctrls), ctrls)).printf()
    return ctrls


def load_last_prompt_handler(*args):
    ctrls = list(args)
    printF(name=MasterName.get_master_name(),
           info="[Parameters] BEFORE--> ctrls: {} - {}".format(len(ctrls), ctrls)).printf()
    if exists(modules.config.last_prompt_path):
        with open(modules.config.last_prompt_path, encoding='utf-8') as json_file:
            try:
                json_obj = json.load(json_file)
                printF(name=MasterName.get_master_name(), info="[Parameters] json_obj = {}".format(json_obj)).printf()
                ctrls = metadata_to_ctrls(json_obj, ctrls)
            except Exception as e:
                printF(name=MasterName.get_master_name(),
                       info="[ERROR] load_last_prompt_handler e = {} -  {} - {}".format(e, len(ctrls), ctrls)).printf()
            finally:
                json_file.close()
    return ctrls


def load_input_images_handler(files):
    return list(map(lambda x: x.name, files)), gr.update(selected=GALLERY_ID_INPUT), gr.update(value=len(files))


def load_revision_images_handler(files):
    return gr.update(value=True), list(map(lambda x: x.name, files[:4])), gr.update(selected=GALLERY_ID_REVISION)


def output_to_input_handler(gallery):
    if len(gallery) == 0:
        return [], gr.update()
    else:
        return list(map(lambda x: x['name'], gallery)), gr.update(selected=GALLERY_ID_INPUT)


def output_to_revision_handler(gallery):
    if len(gallery) == 0:
        return gr.update(value=False), [], gr.update()
    else:
        return gr.update(value=True), list(map(lambda x: x['name'], gallery[:4])), gr.update(
            selected=GALLERY_ID_REVISION)


app = FastAPI()
#
# from fastapi.staticfiles import StaticFiles
# import os
#
# # 确保 static 文件夹存在
# static_dir = os.path.join(os.path.dirname(__file__), "static")
# os.makedirs(static_dir, exist_ok=True)
#
# # 挂载静态文件路由（注意：必须在 mount_gradio_app 之前或之后？通常之前即可）
# app.mount("/static", StaticFiles(directory=static_dir), name="static")

settings = default_settings
reload_javascript()

title = f'MeanVon {main_version}'

if isinstance(adapter.args_manager.args.preset, str):
    title += ' ' + adapter.args_manager.args.preset

def read_ini_nllb(module):
    content = read_ini(module)
    return str(content[0]), int(content[1])


## Functions specific to txt2prompt
def read_ini_txt2prompt(module):
    content = read_ini(module)
    return str(content[0]), int(content[1]), float(content[2]), int(content[3]), int(content[4])


def dump_default_english_config():
    from modules.localization import dump_english_config
    dump_english_config(grh.all_components)

with gr.Blocks(
    title=title,
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.rose,
        secondary_hue=gr.themes.colors.lime,
        neutral_hue=gr.themes.colors.indigo,
        font=["Microsoft YaHei", "微软雅黑", "PingFang SC", "Segoe UI", "sans-serif"]
    ).set(
        body_background_fill="linear-gradient(white 1px, transparent 0), linear-gradient(90deg, white 1px, transparent 0)"
    ),
    css=modules.html.css + """
    /* ===== 只修复宽度自适应，不破坏内部布局 ===== */

    /* 1. 最外层容器铺满 */
    .gradio-container,
    .gradio-container-5-50-0-dev0 {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;      /* 不要用 min-width: 100% */
        margin: 0 auto !important;
        padding: 0 16px !important;
        border-radius: 0 !important;
    }

    /* 2. 内部真正包内容的层级（app -> wrap -> contain）全部放开 max-width */
    .gradio-container main.app,
    .gradio-container .wrap,
    .gradio-container .contain {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
    }

    /* 3. 只让 contain 里的直接 Row 做正常 flex 布局，不要用 100% 硬压 */
    .gradio-container .contain > .block,
    .gradio-container .contain > .form,
    .gradio-container .contain > .group,
    .gradio-container .contain > .row {
        width: 100% !important;
        max-width: 100% !important;
    }

    /* ===== 全局字体：微软雅黑等标准互联网字体 ===== */
    body,
    .gradio-container {
        font-family: 'Microsoft YaHei', '微软雅黑', 'PingFang SC', 'Segoe UI', sans-serif !important;
    }
    """
) as shared.gradio_root:

    currentTask = gr.State(None)

    def init_task():
        """在事件中初始化任务"""
        # 检查是否已初始化
        task = currentTask.value
        if task is None:
            task = task_manager.AsyncTask(args=[])
            return task
        return task

    # 在页面加载时初始化
    shared.gradio_root.load(init_task, None, currentTask)

    with gr.Row():
        with gr.Column():
            with gr.Row():
                nsfw_filter = gr.Radio(label="NSFW Filter", choices=["0", "1"], value="0", visible=True, scale=1,
                                       interactive=True)
            with gr.Row():
                btn_free_gpu_mem = gr.Button(value="Free GPU Memory", size="sm")

            def free_gpu(x):
                free_cuda_mem()
                free_cuda_cache()
                gr.Info("free cuda memory!")
                return gr.update()

            btn_free_gpu_mem.click(fn=free_gpu, inputs=btn_free_gpu_mem, outputs=btn_free_gpu_mem)

            with gr.Accordion(label="Images Viewer", open=False) as finish_image_viewer:
                progress_gallery = gr.Gallery(label='Finished Images', show_label=False, object_fit='contain',
                                              height=700,
                                              visible=False, elem_classes=['main_view'])

            with gr.Row(elem_classes='advanced_check_row'):
                text_factory_checkbox = gr.Checkbox(label='Text-Factory', value=False, container=True,
                                                    info="| Nllb translation | Prompt generator |",
                                                    elem_classes='min_check')


            with gr.Accordion(visible=False) as text_input_panel:
                with gr.Tabs():
                    with gr.Tab("nllb translation 👥", id=15) as tab_nllb:
                        with gr.Accordion("About", open=False):
                            with gr.Group():
                                gr.HTML(
                                    """
                                    <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                    <b>Module : </b>nllb translation</br>
                                    <b>Function : </b>Translate text with <a href='https://ai.meta.com/research/no-language-left-behind/' target='_blank'>nllb</a></br>
                                    <b>Input(s) : </b>Input text</br>
                                    <b>Output(s) : </b>Translated text</br>
                                    <b>HF model page : </b>
                                    <a href='https://huggingface.co/facebook/nllb-200-distilled-600M' target='_blank'>facebook/nllb-200-distilled-600M</a>
                                    </br>
                                    """
                                )
                            with gr.Group():
                                gr.HTML(
                                    """
                                    <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                    <div style='text-align: justified'>
                                    <b>Usage :</b></br>
                                    - Select an <b>input language</b></br>
                                    - Type or copy/paste the text to translate in the <b>source text</b> field</br>
                                    - Select an <b>output language</b></br>
                                    - (optional) modify settings to use another model, or reduce the maximum number of tokens in the output</br>
                                    - Click the <b>Generate</b> button</br>
                                    - After generation, translation is displayed in the <b>Output text</b> field
                                    </div>
                                    """
                                )
                        with gr.Accordion("Settings", open=True):
                            with gr.Row():
                                with gr.Column():
                                    model_nllb = gr.Dropdown(choices=model_list_nllb, value=model_list_nllb[0],
                                                             label="Model",
                                                             info="Choose model to use for inference")
                                with gr.Column():
                                    max_tokens_nllb = gr.Slider(0, 1024, step=1, value=1024, label="Max tokens",
                                                                info="Maximum number of tokens in output")
                            with gr.Row():
                                with gr.Column():
                                    save_ini_btn_nllb = gr.Button("Save custom defaults settings 💾")
                                with gr.Column():
                                    module_name_nllb = gr.Textbox(value="nllb", visible=False, interactive=False)
                                    del_ini_btn_nllb = gr.Button("Delete custom defaults settings 🗑️",
                                                                 interactive=True if test_cfg_exist(
                                                                     module_name_nllb.value) else False)
                                    save_ini_btn_nllb.click(
                                        fn=write_ini,
                                        inputs=[
                                            module_name_nllb,
                                            model_nllb,
                                            max_tokens_nllb,
                                        ]
                                    )
                                    save_ini_btn_nllb.click(fn=lambda: gr.Info('Settings saved'))
                                    save_ini_btn_nllb.click(fn=lambda: gr.update(interactive=True),
                                                            outputs=del_ini_btn_nllb)
                                    del_ini_btn_nllb.click(fn=lambda: del_ini(module_name_nllb.value))
                                    del_ini_btn_nllb.click(fn=lambda: gr.Info('Settings deleted'))
                                    del_ini_btn_nllb.click(fn=lambda: gr.update(interactive=False),
                                                           outputs=del_ini_btn_nllb)
                            if test_cfg_exist(module_name_nllb.value):
                                readcfg_nllb = read_ini_nllb(module_name_nllb.value)
                                model_nllb.value = readcfg_nllb[0]
                                max_tokens_nllb.value = readcfg_nllb[1]
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    source_language_nllb = gr.Dropdown(choices=list(language_list_nllb.keys()),
                                                                       value=list(language_list_nllb.keys())[200],
                                                                       label="Input language",
                                                                       info="Select input language")
                                with gr.Row():
                                    prompt_nllb = gr.Textbox(label="Source text", lines=9, max_lines=9,
                                                             placeholder="Type or paste here the text to translate")
                            with gr.Column():
                                with gr.Row():
                                    output_language_nllb = gr.Dropdown(choices=list(language_list_nllb.keys()),
                                                                       value=list(language_list_nllb.keys())[47],
                                                                       label="Output language",
                                                                       info="Select output language")
                                with gr.Row():
                                    out_nllb = gr.Textbox(label="Output text", lines=9, max_lines=9)

                        with gr.Row():
                            with gr.Column():
                                btn_nllb = gr.Button("Generate 🚀", variant="primary")
                            with gr.Column():
                                btn_nllb_clear_input = gr.ClearButton(components=[prompt_nllb],
                                                                      value="Clear inputs 🧹")
                            with gr.Column():
                                btn_nllb_clear_output = gr.ClearButton(components=[out_nllb],
                                                                       value="Clear outputs 🧹")
                            btn_nllb.click(
                                fn=text_nllb,
                                inputs=[
                                    model_nllb,
                                    max_tokens_nllb,
                                    source_language_nllb,
                                    prompt_nllb,
                                    output_language_nllb,
                                ],
                                outputs=out_nllb,
                                show_progress="full",
                            )

            with gr.Row(elem_classes='advanced_check_row'):
                image_factory_checkbox = gr.Checkbox(label='Image-Factory', value=False, container=True,
                                                     info="| text-to-img | img-to-img |",
                                                     elem_classes='min_check')
                # image_factory_advanced_checkbox = gr.Checkbox(label='Configuration', value=modules.config.default_image_factory_advanced_checkbox, container=True, elem_classes='min_check')
            with gr.Accordion(visible=False) as image_input_panel:
                with gr.Tabs():
                    with gr.Tab(label='Image 2 Image') as uov_tab:
                        with gr.Row():
                            img2img_mode = gr.Checkbox(label='Image Gallery', value=settings['img2img_mode'])
                        with gr.Row(visible=False) as image_2_image_panel:

                            input_gallery = gr.Gallery(label='Input', show_label=True, object_fit='contain',
                                                       height=700,
                                                       visible=True)

                            revision_gallery = gr.Gallery(label='Revision', show_label=True, object_fit='contain',
                                                          height=700, visible=True)
                        with gr.Row():
                            revision_mode = gr.Checkbox(label='Revision (prompting with images)',
                                                        value=settings['revision_mode'])
                        with gr.Row():
                            revision_strength_1 = gr.Slider(label='Revision Strength for Image 1', minimum=-2,
                                                            maximum=2,
                                                            step=0.01,
                                                            value=settings['revision_strength_1'],
                                                            visible=settings['revision_mode'])
                            revision_strength_2 = gr.Slider(label='Revision Strength for Image 2', minimum=-2,
                                                            maximum=2,
                                                            step=0.01,
                                                            value=settings['revision_strength_2'],
                                                            visible=settings['revision_mode'])

                            revision_strength_3 = gr.Slider(label='Revision Strength for Image 3', minimum=-2,
                                                            maximum=2,
                                                            step=0.01,
                                                            value=settings['revision_strength_3'],
                                                            visible=settings['revision_mode'])

                            revision_strength_4 = gr.Slider(label='Revision Strength for Image 4', minimum=-2,
                                                            maximum=2,
                                                            step=0.01,
                                                            value=settings['revision_strength_4'],
                                                            visible=settings['revision_mode'])


                        def revision_changed(value):
                            return gr.update(visible=value == True), gr.update(visible=value == True), gr.update(
                                visible=value == True), gr.update(visible=value == True)

                        with gr.Row():
                            revision_mode.change(fn=revision_changed, inputs=[revision_mode],
                                                 outputs=[revision_strength_1, revision_strength_2, revision_strength_3,
                                                          revision_strength_4])

                            positive_prompt_strength = gr.Slider(label='Positive Prompt Strength', minimum=0, maximum=1,
                                                                 step=0.01,
                                                                 value=settings['positive_prompt_strength'])
                            negative_prompt_strength = gr.Slider(label='Negative Prompt Strength', minimum=0, maximum=1,
                                                                 step=0.01,
                                                                 value=settings['negative_prompt_strength'])

                            img2img_start_step = gr.Slider(label='Image-2-Image Start Step', minimum=0.0, maximum=0.8,
                                                           step=0.01,
                                                           value=settings['img2img_start_step'])
                            img2img_denoise = gr.Slider(label='Image-2-Image Denoise', minimum=0.2, maximum=1.0, step=0.01,
                                                        value=settings['img2img_denoise'])
                            img2img_scale = gr.Slider(label='Image-2-Image Scale', minimum=1.0, maximum=2.0, step=0.25,
                                                      value=settings['img2img_scale'],
                                                      info='For upscaling - use with low denoise values')
                        keep_input_names = gr.Checkbox(label='Keep Input Names', value=settings['keep_input_names'],
                                                       elem_classes='type_small_row')
                        with gr.Row():
                            load_input_images_button = gr.UploadButton(label='Load Image(s) to Input',
                                                                       file_count='multiple',
                                                                       file_types=["image"],
                                                                       elem_classes='type_small_row',
                                                                       min_width=0)
                            load_revision_images_button = gr.UploadButton(label='Load Image(s) to Revision',
                                                                          file_count='multiple', file_types=["image"],
                                                                          elem_classes='type_small_row', min_width=0)
                            output_to_input_button = gr.Button(value='Output to Input',
                                                               elem_classes='type_small_row', min_width=0)
                            output_to_revision_button = gr.Button(value='Output to Revision',
                                                                  elem_classes='type_small_row', min_width=0)

                        img2img_ctrls = [img2img_mode, img2img_start_step, img2img_denoise, img2img_scale,
                                         revision_mode,
                                         positive_prompt_strength, negative_prompt_strength,
                                         revision_strength_1, revision_strength_2, revision_strength_3,
                                         revision_strength_4]


                        def verify_revision(rev, gallery_in, gallery_rev, gallery_out):
                            if rev and len(gallery_rev) == 0:
                                if len(gallery_in) > 0:
                                    gr.Info('Revision: imported input')
                                    return gr.update(), list(map(lambda x: x['name'], gallery_in[:1]))
                                elif len(gallery_out) > 0:
                                    gr.Info('Revision: imported output')
                                    return gr.update(), list(map(lambda x: x['name'], gallery_out[:1]))
                                else:
                                    gr.Warning('Revision: disabled (no images available)')
                                    return gr.update(value=False), gr.update()
                            else:
                                return gr.update(), gr.update()

                        with gr.Row():
                            control_lora_canny = gr.Checkbox(label='Control-LoRA: Canny', value=settings['control_lora_canny'])
                            with gr.Row():
                                canny_edge_low = gr.Slider(label='Edge Detection Low', minimum=0.0, maximum=1.0, step=0.01,
                                                           value=settings['canny_edge_low'], visible=settings['control_lora_canny'])
                                canny_edge_high = gr.Slider(label='Edge Detection High', minimum=0.0, maximum=1.0, step=0.01,
                                                            value=settings['canny_edge_high'],
                                                            visible=settings['control_lora_canny'])
                                canny_start = gr.Slider(label='Canny Start', minimum=0.0, maximum=1.0, step=0.01,
                                                        value=settings['canny_start'], visible=settings['control_lora_canny'])
                                canny_stop = gr.Slider(label='Canny Stop', minimum=0.0, maximum=1.0, step=0.01,
                                                       value=settings['canny_stop'], visible=settings['control_lora_canny'])
                                canny_strength = gr.Slider(label='Canny Strength', minimum=0.0, maximum=2.0, step=0.01,
                                                           value=settings['canny_strength'], visible=settings['control_lora_canny'])


                            def canny_changed(value):
                                return gr.update(visible=value == True), gr.update(visible=value == True), gr.update(
                                    visible=value == True), \
                                    gr.update(visible=value == True), gr.update(visible=value == True)


                            control_lora_canny.change(fn=canny_changed, inputs=[control_lora_canny],
                                                      outputs=[canny_edge_low, canny_edge_high, canny_start, canny_stop,
                                                               canny_strength])
                        with gr.Row():
                            control_lora_depth = gr.Checkbox(label='Control-LoRA: Depth', value=settings['control_lora_depth'])
                            with gr.Row():
                                depth_start = gr.Slider(label='Depth Start', minimum=0.0, maximum=1.0, step=0.01,
                                                        value=settings['depth_start'], visible=settings['control_lora_depth'])
                                depth_stop = gr.Slider(label='Depth Stop', minimum=0.0, maximum=1.0, step=0.01,
                                                       value=settings['depth_stop'], visible=settings['control_lora_depth'])
                                depth_strength = gr.Slider(label='Depth Strength', minimum=0.0, maximum=2.0, step=0.01,
                                                           value=settings['depth_strength'], visible=settings['control_lora_depth'])

                                def depth_changed(value):
                                    return gr.update(visible=value == True), gr.update(visible=value == True), gr.update(
                                        visible=value == True)

                                control_lora_depth.change(fn=depth_changed, inputs=[control_lora_depth],
                                                          outputs=[depth_start, depth_stop, depth_strength])


                    with gr.Tab(label='Upscale or Variation') as uov_tab:
                        with gr.Row():
                            with gr.Column():
                                uov_input_image = grh.Image(label='Drag above image to here', source='upload',
                                                            type='numpy')
                            with gr.Column():
                                uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list,
                                                      value=flags.disabled)
                                gr.HTML(
                                    '<a href="https://github.com/lllyasviel/Fooocus/discussions/390" target="_blank">\U0001F4D4 Document</a>')
                    with gr.Tab(label='ControlNet') as ip_tab:
                        ip_advanced = gr.Checkbox(label='Advanced', value=False, container=False)
                        gr.HTML(
                            '* \"Image Prompt\" <a href="https://github.com/lllyasviel/Fooocus/discussions/557" target="_blank">\U0001F4D4 Document</a>')
                        with gr.Row():
                            ip_images = []
                            ip_types = []
                            ip_stops = []
                            ip_weights = []
                            ip_ctrls = []
                            ip_ad_cols = []
                            for _ in range(flags.controlnet_image_count):
                                with gr.Column():
                                    ip_image = grh.Image(label='Image', source='upload', type='numpy',
                                                         show_label=False,
                                                         height=300)
                                    ip_images.append(ip_image)
                                    ip_ctrls.append(ip_image)
                                    with gr.Column(visible=False) as ad_col:
                                        with gr.Row():
                                            default_end, default_weight = flags.default_parameters[flags.default_ip]

                                            ip_stop = gr.Slider(label='Stop At', minimum=0.0, maximum=1.0,
                                                                step=0.001,
                                                                value=default_end)
                                            ip_stops.append(ip_stop)
                                            ip_ctrls.append(ip_stop)

                                            ip_weight = gr.Slider(label='Weight', minimum=0.0, maximum=2.0,
                                                                  step=0.001,
                                                                  value=default_weight)
                                            ip_weights.append(ip_weight)
                                            ip_ctrls.append(ip_weight)

                                        ip_type = gr.Radio(label='Type', choices=flags.ip_list,
                                                           value=flags.default_ip,
                                                           container=False)
                                        ip_types.append(ip_type)
                                        ip_ctrls.append(ip_type)

                                        ip_type.change(
                                            lambda x: list(
                                                flags.default_parameters[x]
                                            ),  # 或 [flags.default_parameters[x][0], flags.default_parameters[x][1]]
                                            inputs=[ip_type],
                                            outputs=[ip_stop, ip_weight],
                                            queue=False,
                                            show_progress=False,
                                        )

                                    ip_ad_cols.append(ad_col)

                        def ip_advance_checked(x):
                            return (
                                [gr.update(visible=x)] * len(ip_ad_cols)
                                + [gr.update(value=flags.default_ip)] * len(ip_types)
                                + [
                                    gr.update(
                                        value=flags.default_parameters[
                                            flags.default_ip
                                        ][0]
                                    )
                                ]
                                * len(ip_stops)
                                + [
                                    gr.update(
                                        value=flags.default_parameters[
                                            flags.default_ip
                                        ][1]
                                    )
                                ]
                                * len(ip_weights)
                            )


                        ip_advanced.change(ip_advance_checked, inputs=ip_advanced,
                                           outputs=ip_ad_cols + ip_types + ip_stops + ip_weights,
                                           queue=False, show_progress=False)
                    with gr.TabItem(label='Inpaint or Outpaint') as inpaint_tab:
                        with gr.Row():
                            inpaint_input_image = grh.Image(label='Drag inpaint or outpaint image to here',
                                                            source='upload', type='numpy', tool='sketch',
                                                            height=500,
                                                            brush_color="#FFFFFF", elem_id='inpaint_canvas')
                            inpaint_mask_image = grh.Image(label='Mask Upload', source='upload', type='numpy',
                                                           height=500, visible=False)

                        with gr.Row():
                            outpaint_expansion_ratio = gr.Slider(label='Outpaint Expansion Ratio', minimum=0.05,
                                                                 maximum=1.3, step=0.05, value=0.3)
                        with gr.Row():
                            inpaint_additional_prompt = gr.Textbox(placeholder="Describe what you want to inpaint.",
                                                                   elem_id='inpaint_additional_prompt',
                                                                   label='Inpaint Additional Prompt', visible=False)
                            outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'],
                                                                   value=[],
                                                                   label='Outpaint Direction')
                            inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options,
                                                       value=modules.flags.inpaint_option_default, label='Method')
                        example_inpaint_prompts = gr.Dataset(samples=modules.config.example_inpaint_prompts,
                                                             label='Additional Prompt Quick List',
                                                             components=[inpaint_additional_prompt], visible=False)
                        gr.HTML(
                            '* Powered by Fooocus Inpaint Engine <a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">\U0001F4D4 Document</a>')
                        example_inpaint_prompts.click(lambda x: x[0], inputs=example_inpaint_prompts,
                                                      outputs=inpaint_additional_prompt, show_progress=False,
                                                      queue=False)
                    with gr.TabItem(label='Describe') as desc_tab:
                        with gr.Row():
                            with gr.Column():
                                desc_input_image = grh.Image(label='Drag any image to here', source='upload',
                                                             type='numpy')
                            with gr.Column():
                                desc_method = gr.Radio(
                                    label='Content Type',
                                    choices=[flags.desc_type_photo, flags.desc_type_anime],
                                    value=flags.desc_type_photo)
                                desc_btn = gr.Button(value='Describe this Image into Prompt')
                                gr.HTML(
                                    '<a href="https://github.com/lllyasviel/Fooocus/discussions/1363" target="_blank">\U0001F4D4 Document</a>')

                    with gr.TabItem(label='Metadata') as load_tab:
                        with gr.Column():
                            metadata_input_image = grh.Image(label='Drag any image generated by Fooocus here',
                                                             source='upload', type='filepath')
                            metadata_json = gr.JSON(label='Metadata')
                            metadata_import_button = gr.Button(value='Apply Metadata')


                        def trigger_metadata_preview(filepath):
                            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(filepath)

                            results = {}
                            if parameters is not None:
                                results['parameters'] = parameters

                            if isinstance(metadata_scheme, flags.MetadataScheme):
                                results['metadata_scheme'] = metadata_scheme.value

                            return results

                        metadata_input_image.upload(trigger_metadata_preview, inputs=metadata_input_image,
                                                    outputs=metadata_json, queue=False, show_progress=True)


            def is_model_imported(model_name):
                # print(f"sys.modules: {sys.modules}")
                return model_name in sys.modules.keys()


            with gr.Row(elem_classes='advanced_check_row'):
                video_factory_checkbox = gr.Checkbox(label='Video-Factory', value=False, container=True,
                                                     info="| img-to-vid |",
                                                     elem_classes='min_check')
                # video_factory_advanced_checkbox = gr.Checkbox(label='Configuration', value=modules.config.default_video_factory_advanced_checkbox, container=True, elem_classes='min_check')

            with gr.Accordion(visible=False) as video_input_panel:
                from modules.config import svd_config

                version = svd_config.get("version")
                default_fps = infer_args.get('fps', 1)

                with gr.Tabs():

                    if ram_size() >= 16:
                        titletab_tab_svd = "Stable Video Diffusion 📼"
                    else:
                        titletab_tab_svd = "Stable Video Diffusion ⛔"
                    with gr.Tab(titletab_tab_svd, id=151) as tab_svd:
                        with gr.Group():   # 不能嵌套 gr.Blocks
                            with gr.Row():
                                image = gr.Image(label="input image", type="filepath", elem_id='img-box')
                                video_out = gr.Video(label="generated video", elem_id='video-box')
                            with gr.Column():
                                model_load_flag = gr.Checkbox(label="set flag to model loader", value=False,
                                                              info="if checked, force to load svd model")
                                resize_image = gr.Checkbox(label="resize to optimal size", value=True)
                                btn = gr.Button("Run")
                                with gr.Accordion(label="Advanced options", open=False):
                                    with gr.Row():
                                        n_frames = gr.Number(precision=0, label="number of frames",
                                                             value=svd_config.get(version).get("num_frames"))
                                        n_steps = gr.Number(precision=0, label="number of steps",
                                                            value=svd_config.get(version).get("num_steps"))
                                        seed = gr.Textbox(value="random", label="seed (integer or 'random')")
                                    with gr.Row():
                                        decoding_t = gr.Number(precision=0,
                                                               label="number of frames decoded at a time",
                                                               value=1)
                                        fps_id = gr.Number(precision=0, label="frames per second",
                                                           value=default_fps)
                                        motion_bucket_id = gr.Number(precision=0, value=127,
                                                                     label="motion bucket id")
                                    with gr.Row():
                                        cond_aug = gr.Number(label="condition augmentation factor", value=0.02)

                            examples = [[
                                "sdxl_styles_samples/Fooocus V2.png",
                                False, True,
                                svd_config.get(version).get("num_frames"),
                                svd_config.get(version).get("num_steps"),
                                "random", 1, default_fps, 127, 0.02
                            ]]
                            inputs = [image, model_load_flag, resize_image, n_frames, n_steps, seed, decoding_t,
                                      fps_id, motion_bucket_id, cond_aug]
                            outputs = [video_out]

                            free_cuda_mem()
                            free_cuda_cache()
                            from modules.model import infer

                            btn.click(infer, inputs=inputs, outputs=outputs)
                            gr.Examples(examples=examples, inputs=inputs, outputs=outputs, fn=infer)

            ip_advanced.change(lambda: None, queue=False, show_progress=False, js=down_js)

            current_tab = gr.State(value='uov')
            # current_tab = gr.Textbox(value='uov', visible=False)
            # default_image = gr.State(value=None)

            # lambda_img = lambda x: x['image'] if isinstance(x, dict) else x
            # uov_input_image.upload(lambda_img, inputs=uov_input_image, outputs=default_image, queue=False)
            # inpaint_input_image.upload(lambda_img, inputs=inpaint_input_image, outputs=default_image, queue=False)

            def update_video_factory(x):
                return gr.update(visible=x), gr.update(visible=x)


            def update_image_factory(x):
                return gr.update(visible=x)


            def update_text_factory(x):
                return gr.update(visible=x)


            image_factory_checkbox.change(update_image_factory, inputs=image_factory_checkbox,
                                          outputs=image_input_panel, queue=False, show_progress=False, js=switch_js)

            video_factory_checkbox.change(update_video_factory, inputs=video_factory_checkbox,
                                          outputs=[video_input_panel, btn], queue=False, show_progress=False, js=switch_js)

            text_factory_checkbox.change(update_text_factory, inputs=text_factory_checkbox,
                                         outputs=text_input_panel, queue=False, show_progress=False, js=switch_js)



            def update_default_image(x):
                global default_image
                if isinstance(x, dict):
                    default_image = x['image']
                else:
                    default_image = x
                return


            def clear_default_image():
                global default_image
                default_image = None
                return


            uov_tab.select(lambda: 'uov', outputs=current_tab, queue=False, js=down_js, show_progress=False)
            inpaint_tab.select(lambda: 'inpaint', outputs=current_tab, queue=False, js=down_js,
                               show_progress=False)
            ip_tab.select(lambda: 'ip', outputs=current_tab, queue=False, js=down_js, show_progress=False)
            desc_tab.select(lambda: 'desc', outputs=current_tab, queue=False, js=down_js, show_progress=False)

        with gr.Column(scale=1):
            progress_window = grh.Image(label='Preview', show_label=True, height=640, visible=False)
            progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False,
                                    elem_id='progress-bar', elem_classes='progress-bar')
            # gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain', height=720, visible=True, elem_classes='resizable_area')
            with gr.Column() as gallery_holder:
                with gr.Tabs(selected=GALLERY_ID_OUTPUT) as gallery_tabs:
                    with gr.Tab(label='Output', id=GALLERY_ID_OUTPUT):
                        output_gallery = gr.Gallery(label='Output', show_label=False, object_fit='contain',
                                                    height=700,
                                                    visible=True)
                    # with gr.Tab(label='Gallery', id=GALLERY_ID_FINISH):
                    #     gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain', height=745,
                    #                          visible=True, elem_classes='resizable_area')

            with gr.Row():
                remain_images_progress = gr.Textbox(label="Images process progress", value=settings["image_number"],
                                                    elem_classes='type_row_spec', visible=False, show_label=False)
            with gr.Row():
                generate_button = gr.Button(value="Generate", elem_classes='type_row_half',
                                            elem_id='generate_button', visible=True)
                # load_parameter_button = gr.Button(value="Load Parameters",
                #                                   elem_classes='type_row', elem_id='load_parameter_button',
                #                                   visible=False)
                skip_button = gr.Button(value="Skip", elem_classes='type_row_half', visible=False)
                stop_button = gr.Button(value="Stop", elem_classes='type_row_half',
                                        elem_id='stop_button', visible=False)


                def stop_clicked(currentTask):
                    import ldm_patched.modules.model_management as model_management
                    currentTask.last_stop = 'stop'
                    if currentTask.processing:
                        model_management.interrupt_current_processing()
                    return currentTask


                def skip_clicked(currentTask):
                    import ldm_patched.modules.model_management as model_management
                    currentTask.last_stop = 'skip'
                    if currentTask.processing:
                        model_management.interrupt_current_processing()
                    return currentTask

                # 新版本：去掉 every, queue, show_progress；js 参数保持不变（无下划线）
                stop_button.click(stop_clicked, inputs=currentTask, outputs=currentTask)
                skip_button.click(skip_clicked, inputs=currentTask, outputs=currentTask)

            with gr.Row(elem_classes='prompt_row'):
                prompt = gr.Textbox(
                    label='Prompt', show_label=True,
                    placeholder="Type prompt here.",
                    container=True, autofocus=True,
                    elem_classes='prompt_row',
                    elem_id="main_prompt",          # <-- 添加
                    lines=5, max_lines=20,
                    info='Describing objects that you DO want to see.',
                    value=settings['prompt']
                )
            with gr.Row(elem_classes='n_negative_prompt_row'):
                negative_prompt = gr.Textbox(
                    label='Negative Prompt', show_label=True,
                    placeholder="Type negative prompt here.",
                    elem_id="negative_prompt",      # <-- 添加
                    info='Describing objects that you DO NOT want to see.',
                    lines=5, max_lines=20
                )

        with gr.Column(scale=1, visible=settings['advanced_mode']) as advanced_column:
            with gr.Tab(label='Setting'):
                # fooocus code, ignore begin
                # if not adapter.args_manager.args.disable_preset_selection:
                #     preset_selection = gr.Radio(label='Preset',
                #                                 choices=modules.config.available_presets,
                #                                 value=adapter.args_manager.args.preset if adapter.args_manager.args.preset else "initial",
                #                                 interactive=True)
                # fooocus code, ignore end
                performance_selection = gr.Radio(label='Performance',
                                                 choices=flags.Performance.list(),
                                                 value=modules.config.default_performance)
                with gr.Row():
                    fixed_steps = gr.Slider(visible=True, label='Steps', minimum=1, maximum=200, step=1,
                                            value=settings['fixed_steps'])
                with gr.Row(visible=settings['performance'] == 'Custom') as custom_row:
                    custom_steps = gr.Slider(label='Custom Steps', minimum=1, maximum=200, step=1,
                                             value=settings['custom_steps'])
                    custom_switch = gr.Slider(label='Custom Switch', minimum=0.2, maximum=1.0, step=0.01,
                                              value=settings['custom_switch'])
                aspect_ratios_selection = gr.Radio(label='Aspect Ratios',
                                                   choices=modules.config.available_aspect_ratios,
                                                   value=modules.config.default_aspect_ratio, info='width × height',
                                                   elem_classes='aspect_ratios')

                image_number = gr.Slider(label='Image Number', minimum=1,
                                         maximum=modules.config.default_max_image_number, step=1,
                                         value=modules.config.default_image_number)

                with gr.Row():
                    seed_random = gr.Checkbox(label='Random seed', value=settings['seed_random'])
                    same_seed_for_all = gr.Checkbox(label='Same seed for all images',
                                                    value=settings['same_seed_for_all'])
                image_seed = gr.Textbox(label='Seed', value=settings['seed'], max_lines=1,
                                        visible=not settings['seed_random'])

                with gr.Row():
                    play_notification_sound = gr.Checkbox(label='Notificate me when all tasks done',
                                                          value=settings['play_notification_sound'], interactive=True)

                    notification_file = 'notification.mp3'
                    if os.path.exists(notification_file):
                        notification = gr.State(value=notification_file)
                        notification_input = gr.Audio(label='Notification', interactive=True,
                                                      value=notification_file,
                                                      elem_id='audio_notification', visible=settings['play_notification_sound'],
                                                      )

                    def play_notification_checked(r, notification):
                        return gr.update(visible=r, value=notification if r else None)


                    def notification_input_changed(notification_input, notification):
                        if notification_input:
                            notification = notification_input
                        return notification


                    play_notification_sound.change(fn=play_notification_checked,
                                                   inputs=[play_notification_sound, notification],
                                                   outputs=[notification_input], queue=False)
                    notification_input.change(fn=notification_input_changed,
                                              inputs=[notification_input, notification], outputs=[notification],
                                              queue=False)

                def get_scope_of_influence():
                    return '<b>Valid Saved Parameters (as below):</b>' \
                        + ' <br> <font color="blue" size="1">-->Prompt, Negative Prompt,  Performance, Custom Steps, Aspect Ratios, Image Seed, </font>' \
                        + ' <br> <font color="blue" size="1">-->Final Style Keys, </font>' \
                        + ' <br> <font color="blue" size="1">-->Base Model, Refiner, Refiner Switch, LoRAs,</font>' \
                        + ' <br> <font color="blue" size="1">-->Base CLIP Skip, Refiner CLIP Skip, Sharpness, Guidance Scale, Sampler, Scheduler.</font>' \
                        + ' '


                scope_of_influence = gr.HTML(value=get_scope_of_influence())
                with gr.Row():
                    load_prompt_button = gr.UploadButton(label='Load Prompt', file_count='single',
                                                         file_types=['.json', '.png', '.jpg', '.webp'],
                                                         elem_classes='type_small_row', min_width=0)
                    load_last_prompt_button = gr.Button(value='Load Last Prompt',
                                                        elem_classes='type_small_row', min_width=0)


                def get_current_links():
                    return '<b>links:</b>' \
                        + '<br>-wiki: <a href="https://github.com/lllyasviel/Fooocus/discussions/117">&#128212; Fooocus Advanced</a>' \
                        + ' <a href="https://github.com/MoonRide303/Fooocus-MRE/wiki">&#128212; Fooocus-MRE Wiki</a>' \
                        + '<br><br><b>Logs:</b><br>' \
                        + f' <a href="/file={get_previous_log_path()}" target="_blank">\U0001F4DA Yesterday Log</a>'


                links = gr.HTML(value=get_current_links())


                def random_checked(r):
                    return gr.update(visible=not r)


                def refresh_seed(r, seed_string):
                    if r:
                        return random.randint(constants.MIN_SEED, constants.MAX_SEED)
                    else:
                        try:
                            seed_value = int(seed_string)
                            if constants.MIN_SEED <= seed_value <= constants.MAX_SEED:
                                return seed_value
                        except ValueError:
                            pass
                        return random.randint(constants.MIN_SEED, constants.MAX_SEED)


                seed_random.change(random_checked, inputs=[seed_random], outputs=[image_seed],
                                   queue=False, show_progress=False)


                def update_history_link():
                    if adapter.args_manager.args.disable_image_log:
                        return gr.update(value='')

                    return gr.update(
                        value=f'<a href="file={get_current_html_path()}" target="_blank">\U0001F4DA Today Log</a>')


                history_link = gr.HTML()
                shared.gradio_root.load(update_history_link, outputs=[history_link], queue=False, show_progress="full")


                def performance_changed(ps, fs):
                    if ps == "Custom":
                        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True,
                                                                                            value=fs), gr.update(
                            visible=True)
                    elif ps == "Speed":
                        return gr.update(visible=True, label="Fixed Steps", value=constants.STEPS_SPEED,
                                         interactive=False), gr.update(visible=False), gr.update(
                            visible=False), gr.update(visible=False)
                    elif ps == "Quality":
                        return gr.update(visible=True, label="Fixed Steps", value=constants.STEPS_QUALITY,
                                         interactive=False), gr.update(visible=False), gr.update(
                            visible=False), gr.update(visible=False)
                    elif ps == "Lightning":
                        if fs is None or fs == "" or int(fs) < 1 or int(fs) > 8:
                            _v = constants.STEPS_LIGHTNING
                        else:
                            _v = fs
                        return gr.update(visible=True, label="Fixed Steps", value=_v,
                                         interactive=True), gr.update(visible=False), gr.update(
                            visible=False), gr.update(visible=False)
                    elif ps == "LCM":
                        if fs is None or fs == "" or int(fs) < 1 or int(fs) > 8:
                            _v = constants.STEPS_LCM
                        else:
                            _v = fs
                        return gr.update(visible=True, label="Fixed Steps", value=_v,
                                         interactive=True), gr.update(visible=False), gr.update(
                            visible=False), gr.update(visible=False)
                    elif ps == "TURBO":
                        if fs is None or fs == "" or int(fs) < 1 or int(fs) > 8:
                            _v = constants.STEPS_TURBO
                        else:
                            _v = fs
                        return gr.update(visible=True, label="Fixed Steps", value=_v,
                                         interactive=True), gr.update(visible=False), gr.update(
                            visible=False), gr.update(visible=False)
                    elif ps == "Hyper-SD":
                        if fs is None or fs == "" or int(fs) < 1 or int(fs) > 4:
                            _v = constants.STEPS_HYPER_SD
                        else:
                            _v = fs
                        return gr.update(visible=True, label="Fixed Steps", value=_v,
                                         interactive=True), gr.update(visible=False), gr.update(
                            visible=False), gr.update(visible=False)


                performance_selection.change(fn=performance_changed, inputs=[performance_selection, fixed_steps],
                                             outputs=[fixed_steps, custom_row, custom_steps, custom_switch])


                def style_iterator_changed(_style_iterator, _style_selections):
                    if _style_iterator:
                        combinations_count = 1 + len(style_keys) - len(
                            _style_selections)  # original style selection + all remaining style combinations
                        return gr.update(interactive=False, value=combinations_count)
                    else:
                        return gr.update(interactive=True, value=settings['image_number'])


                gr.HTML('<b>Github:</b><br>' \
                        + ' <a href="https://github.com/teshu2you/Meanvon" target="_blank">\U0001F4DA  Meanvon</a>' \
                        + '<br><a href="https://github.com/lllyasviel/Fooocus" target="_blank">\U0001F4DA  Fooocus</a>' \
                        + ' <a href="https://github.com/MoonRide303/Fooocus-MRE" target="_blank">\U0001F4DA  Fooocus-MRE</a>' \
                        + ' <a href="https://github.com/mrhan1993/Fooocus-API" target="_blank">\U0001F4DA  Fooocus-API</a>' \
                        + ' <a href="https://github.com/runew0lf/RuinedFooocus" target="_blank">\U0001F4DA  RuinedFooocus</a>' \
                        + '<br> <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui" target="_blank">\U0001F4DA  stable-diffusion-webui</a> ' \
                        + ' <a href="https://github.com/lllyasviel/stable-diffusion-webui-forge" target="_blank">\U0001F4DA  stable-diffusion-webui-forge</a> ' \
                        + '<br> <a href="https://github.com/comfyanonymous/ComfyUI" target="_blank">\U0001F4DA  ComfyUI</a>' \
                        + '<br> <a href="https://github.com/Woolverine94/biniou/" target="_blank">\U0001F4DA  biniou</a>' \
                        + '<br> <a href="https://github.com/CCmahua/ChatTTS-Enhanced" target="_blank">\U0001F4DA  ChatTTS-Enhanced</a>'
                        )


            with gr.Tab(label='Style'):
                style_class = gr.Radio(label='Style Selector',
                                       choices=['Default', 'ALL_Checked', 'ALL_UnChecked'] + list(
                                           hot_style_keys.keys()),
                                       value='Default', interactive=True, show_label=True)

                style_result = gr.Textbox(label="Final Style Keys",
                                          value=modules.config.default_styles, lines=5, max_lines=100,
                                          visible=True)
                with gr.Accordion(label='Style Sample', open=False) as style_sample:
                    style_gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain', height=300,
                                               visible=True, elem_classes='resizable_area')

                with gr.Accordion(label='Style keys List', open=True):
                    style_sorter.try_load_sorted_styles(
                        style_names=legal_style_names,
                        default_selected=modules.config.default_styles)

                    style_search_bar = gr.Textbox(show_label=False, container=False,
                                                  placeholder="\U0001F50E Type here to search styles ...",
                                                  value="",
                                                  label='Search Styles')
                    style_selections = gr.CheckboxGroup(show_label=False, container=False,
                                                        choices=copy.deepcopy(style_sorter.all_styles),
                                                        value=copy.deepcopy(modules.config.default_styles),
                                                        label='Selected Styles',
                                                        elem_classes=['style_selections'])
                    gradio_receiver_style_selections = gr.Textbox(elem_id='gradio_receiver_style_selections',
                                                                  visible=False)

                    shared.gradio_root.load(lambda: gr.update(choices=copy.deepcopy(style_sorter.all_styles)),
                                            outputs=style_selections)

                    style_search_bar.change(style_sorter.search_styles,
                                            inputs=[style_selections, style_search_bar],
                                            outputs=style_selections,
                                            queue=False,
                                            show_progress=False).then(
                        lambda: None, js='refresh_style_localization')

                    gradio_receiver_style_selections.input(style_sorter.sort_styles,
                                                           inputs=style_selections,
                                                           outputs=style_selections,
                                                           queue=False,
                                                           show_progress=False).then(
                        lambda: None, js='refresh_style_localization')


                    def show_sub_style_selection(ssk: dict, nn, op=0):
                        """
                        op
                             none: ignore
                             1: all checked
                             0: all unchecked
                        """
                        sss = []
                        for k, v in ssk.items():
                            li = list(v)
                            num = str(len(li))
                            if k == "" and v == []:
                                k = "In total"
                                num = str(nn)
                            if k == "Default":
                                continue
                            ch = li
                            vl = [fooocus_expansion]
                            if op is None:
                                vl = legal_style_names
                            elif 0 == op:
                                vl = []
                            elif 1 == op:
                                vl += li
                            elif 2 == op:
                                vl += li
                            sss.append(gr.CheckboxGroup(show_label=True, container=True,
                                                        choices=ch,
                                                        value=vl,
                                                        label=str(k) + ": (" + num + ")"))
                            print(num)
                        print(sss)
                        return sss


                    sub_style_keys = {}
                    n = 1
                    _sk_old = ""

                    _style_keys = sorted(style_keys, key=str.lower)
                    all_style_keys = _style_keys
                    STYLE_NUM = len(_style_keys)

                    while n <= STYLE_NUM:
                        _sub_style_keys = []
                        for sk in _style_keys:
                            _sk_new = sk.split(" ")[0]
                            if _sk_new == _sk_old:
                                _sub_style_keys.append(sk)
                            else:
                                sub_style_keys.update({_sk_old: _sub_style_keys})
                                _sub_style_keys = [sk]
                                sub_style_keys.update({_sk_new: _sub_style_keys})
                            _sk_old = _sk_new
                            n += 1


                    def change_style_class(choice):
                        def _p(s):
                            tmp = hot_style_keys.get(s, [])
                            printF(name=MasterName.get_master_name(),
                                   info="[info] hot_style_keys = {}".format(tmp)).printf()
                            tmp = [normalize_key(x) for x in tmp]
                            return tmp

                        print(choice)
                        if choice == "Default":
                            return gr.update(show_label=False, container=False,
                                                           choices=legal_style_names,
                                                           value=default_legal_style_names,
                                                           label='Image Style')
                        elif choice == "ALL_Checked":
                            return gr.update(show_label=False, container=False,
                                                           choices=legal_style_names,
                                                           value=legal_style_names,
                                                           label='Image Style')
                        elif choice == "ALL_UnChecked":
                            return gr.update(show_label=False, container=False,
                                                           choices=legal_style_names,
                                                           value=[],
                                                           label='Image Style')
                        else:
                            special = _p(choice)
                            return gr.update(show_label=False, container=False,
                                                           choices=[fooocus_expansion] + special,
                                                           value=special,
                                                           label='Image Style')


                    style_class.change(fn=change_style_class, inputs=style_class, outputs=style_selections)


                def _convert_style(x):
                    path = modules.config.path_style_samples + "\\"
                    default_file = (path + "blank_style.png", "blank_style")
                    _files = []
                    file_subfix = [".jpg", ".jpeg", ".png", ".webp", ".tiff", ".jp2"]
                    for xx in x:
                        for file in [f'{path}{xx}{ext}' for ext in file_subfix]:
                            if os.path.isfile(file):
                                printF(name=MasterName.get_master_name(), info="[info] file = {}".format(file)).printf()
                                file = (file, xx)
                                _files.append(file)
                    if _files:
                        return [x.__str__(), gr.update(open=True), gr.update(value=_files)]
                    return [x.__str__(), gr.update(open=False), gr.update(value=[default_file])]


                style_selections.change(fn=_convert_style,
                                        inputs=style_selections,
                                        outputs=[style_result, style_sample, style_gallery])

            with gr.Tab(label='Model'):
                def get_target_info(x=""):
                    json_obj = {
                        "Type": "",
                        "NSFW": "",
                        "Update_Date": "",
                        "Base": "",
                        "Links": "",
                        "Version": "",
                        "Tags": "",
                        "Usage_Tips": "",
                        "Author": "",
                        "ReMark": "",
                    }

                    if x is not None and "lora" in x.lower():
                        path = modules.config.paths_loras[0]
                    else:
                        path = modules.config.paths_checkpoints[0]

                    # 使用 os.path.join 处理路径，避免手动拼接反斜杠
                    target_name = os.path.join(path, x + ".json")

                    if os.path.isfile(target_name):
                        try:
                            with open(target_name, encoding="utf-8") as json_file:
                                json_obj = json.load(json_file)
                                printF(
                                    name=MasterName.get_master_name(),
                                    info="[Parameters] json_obj = {}".format(json_obj),
                                ).printf()
                        except Exception as e:
                            printF(
                                name=MasterName.get_master_name(),
                                info="json -- get_target_info, e: {}".format(e),
                            ).printf()
                        # finally 块不需要了，with 上下文管理器会自动关闭文件

                    return json_obj


                def convert_json_to_html(x):
                    _html = ""
                    for kk, vv in x.items():
                        if "http" in vv:
                            _html += '<b>' + kk + '</b>: <br>' + '<font color="blue" size="1"> <a href=' + vv + ' target="_blank">' + vv + '</a></font> <br>'
                            continue
                        _html += '<b>' + kk + '</b>: <br>' + '<font color="blue" size="1">' + vv + '</font> <br>'
                    return _html


                with gr.Row():
                    model_presets = gr.Radio(label="Model Preset Selector", show_label=True, container=True,
                                             choices=modules.config.preset_filenames,
                                             value=modules.config.default_model_preset_name,
                                             info='Scope of influence:[Prompt、Aspect Ratios、Style Keys、Advanced]')
                with gr.Row():
                    model_type_selector = gr.Dropdown(label='Model Type Selector',
                                                      choices=modules.config.model_types,
                                                      value=modules.config.default_model_type, show_label=True)
                with gr.Row():
                    base_model = gr.Dropdown(
                    label='Base Model (SDXL only)',
                    choices=modules.config.sd_model_filenames,
                    value=modules.config.default_base_model_name,
                    show_label=True,
                    allow_custom_value=True,
                )

                    refiner_model = gr.Dropdown(
                        label='Refiner (SDXL or SD 1.5)',
                        choices=['None'] + modules.config.model_filenames,
                        value=modules.config.default_refiner_model_name,
                        show_label=True,
                        allow_custom_value=True,
                    )
                with gr.Row():
                    with gr.Accordion(label="-", open=False) as bm_acc:
                        img_bm_thumbnail = grh.Image(label='bm_thumbnail', type='filepath', show_label=False,
                                                     height=300)
                        with gr.Accordion(label="--", open=False):
                            img_bm_info = gr.HTML(value="")

                    with gr.Accordion(label="-", open=False) as rm_acc:
                        img_rm_thumbnail = grh.Image(label='rm_thumbnail', type='filepath', show_label=False,
                                                     height=300)
                        with gr.Accordion(label="--", open=False):
                            img_rm_info = gr.HTML(value="")

                refiner_switch = gr.Slider(label='Refiner Switch At', minimum=0.1, maximum=1.0, step=0.0001,
                                           info='Use 0.4 for SD1.5 realistic models; '
                                                'or 0.667 for SD1.5 anime models; '
                                                'or 0.8 for XL-refiners; '
                                                'or any value for switching two SDXL models.',
                                           value=modules.config.default_refiner_switch,
                                           visible=modules.config.default_refiner_model_name not in ['None',
                                                                                                     'Not Exist!->'])


                def get_thumbnail_info(x):
                    if x is None:
                        x = "None"
                    if x is not None and "lora" in x.lower():
                        path = modules.config.paths_loras[0] + "\\"
                    else:
                        path = modules.config.paths_checkpoints[0] + "\\"
                    printF(name=MasterName.get_master_name(),
                           info="[Parameters] path,x = {}{}".format(path, x)).printf()

                    file_subfix = [".jpg", ".jpeg", ".png", ".webp", ".tiff", ".jp2"]
                    if "." in x:
                        x = x.split(".")[:-1]
                        x = ".".join(x)
                        for file in [f'{path}{x}{ext}' for ext in file_subfix]:
                            if os.path.isfile(file):
                                # print(f"file:{file}")
                                info = convert_json_to_html(get_target_info(x))
                                return [gr.update(open=True), gr.update(value=file), gr.update(value=info)]
                    return [gr.update(open=False), gr.update(value=None), gr.update(value="")]


                def get_model_type_selector(model_type, current_model):
                    model_type_key = str(model_type or "").lower()

                    model_choices = ["None"]
                    for model_type_name in modules.config.model_types:
                        if model_type_key in str(model_type_name).lower():
                            model_choices += modules.config.get_model_filenames(
                                modules.config.modelfile_path,
                                name_filter=model_type_name,
                            )

                    model_choices = list(dict.fromkeys(model_choices))
                    model_value = (
                        current_model
                        if current_model in model_choices
                        else model_choices[0]
                    )

                    return [
                        gr.update(
                            label=model_type,
                            choices=model_choices,
                            value=model_value,
                        ),
                        gr.update(
                            choices=["None"] + list(modules.config.model_filenames),
                            value="None",
                            visible=True,
                        ),
                        gr.update(visible=True),
                    ]


                model_type_selector.change(fn=get_model_type_selector, inputs=[model_type_selector, base_model],
                                           outputs=[base_model, refiner_model, style_class],
                                           show_progress=False, queue=False)  \
                .then(fn=change_style_class, inputs=style_class, outputs=style_selections)

                base_model.change(fn=get_thumbnail_info, inputs=base_model,
                                  outputs=[bm_acc, img_bm_thumbnail, img_bm_info],
                                  show_progress=False, queue=False)
                refiner_model.change(fn=get_thumbnail_info, inputs=refiner_model,
                                     outputs=[rm_acc, img_rm_thumbnail, img_rm_info],
                                     show_progress=False, queue=False)


                def adjust_ref_switch(x):
                    y = 0.8
                    if x is None:
                        x = "None"
                    if not any(_x in x.upper() for _x in ["SDXL", "SD_XL", "XL"]):
                        if "realistic" in x.lower():
                            y = 0.4
                        elif "anime" in x.lower():
                            y = 0.667
                    return gr.update(visible=x not in ['None', 'Not Exist!->'], value=y)


                refiner_model.change(adjust_ref_switch,
                                     inputs=refiner_model, outputs=refiner_switch, show_progress=False, queue=False)

                with gr.Group():
                    lora_ctrls = []
                    for i, (enabled, filename, weight) in enumerate(modules.config.default_loras):
                        with gr.Row():
                            lora_enabled = gr.Checkbox(label='Enable', value=enabled,
                                                       elem_classes=['lora_enable', 'min_check'], scale=1,
                                                       show_label=True)
                            lora_model = gr.Dropdown(label=f'LoRA {i + 1}',
                                                     choices=['None'] + modules.config.lora_filenames, value=filename,
                                                     elem_classes='lora_model', scale=10)
                            lora_weight = gr.Slider(label='Weight', minimum=modules.config.default_loras_min_weight,
                                                    maximum=modules.config.default_loras_max_weight, step=0.01,
                                                    value=weight,
                                                    elem_classes='lora_weight', scale=5)

                        with gr.Row():
                            with gr.Accordion(label="-", open=False) as lora_acc:
                                img_lora_thumbnail = grh.Image(label='lora_thumbnail', type='filepath',
                                                               show_label=False, height=300)
                                with gr.Accordion(label="--", open=False):
                                    img_lora_info = gr.HTML(value="")

                            lora_model.change(fn=get_thumbnail_info, inputs=lora_model,
                                              outputs=[lora_acc, img_lora_thumbnail, img_lora_info],
                                              show_progress=False, queue=False)

                            lora_ctrls += [lora_enabled, lora_model, lora_weight]
                with gr.Row():
                    canny_model = gr.Dropdown(label='Canny Model',
                                              choices=modules.config.controlnet_lora_canny_filenames,
                                              value=modules.config.default_controlnet_canny_name)
                    depth_model = gr.Dropdown(label='Depth Model',
                                              choices=modules.config.controlnet_lora_depth_filenames,
                                              value=modules.config.default_controlnet_depth_name)
                with gr.Row():
                    model_refresh = gr.Button(value='\U0001f504 Refresh All Files',
                                              variant='secondary', elem_classes='refresh_button')

                with gr.Row():
                    with gr.Accordion(label="Remark", open=False):
                        model_lora_remark = gr.Textbox(label="tips about model and lora", show_label=True,
                                                       value="remark", container=True, lines=10, max_lines=1024)

                canny_ctrls = [control_lora_canny, canny_edge_low, canny_edge_high, canny_start, canny_stop,
                               canny_strength, canny_model]
                depth_ctrls = [control_lora_depth, depth_start, depth_stop, depth_strength, depth_model]

            with gr.Tab(label='Advanced'):
                with gr.Row():
                    base_clip_skip = gr.Slider(label='Base CLIP Skip', minimum=-10, maximum=-1, step=1,
                                               value=settings['base_clip_skip'])
                    refiner_clip_skip = gr.Slider(label='Refiner CLIP Skip', minimum=-10, maximum=-1, step=1,
                                                  value=settings['refiner_clip_skip'])
                sharpness = gr.Slider(label='Image Sharpness', minimum=0.0, maximum=30.0, step=0.001,
                                      value=modules.config.default_sample_sharpness,
                                      info='Higher value means image and texture are sharper.')
                guidance_scale = gr.Slider(label='Guidance Scale', minimum=1.0, maximum=30.0, step=0.01,
                                           value=modules.config.default_cfg_scale,
                                           info='Higher value means style is cleaner, vivider, and more artistic.')

                gr.HTML(
                    '<a href="https://github.com/lllyasviel/Fooocus/discussions/117" target="_blank">\U0001F4D4 Document</a>')

                dev_mode = gr.Checkbox(label='Developer Debug Mode', value=False, container=False)

                with gr.Column(visible=False) as dev_tools:
                    with gr.Tab(label='Debug Tools'):
                        disable_preview = gr.Checkbox(label='Disable Preview', value=False,
                                                      info='Disable preview during generation.')

                        generate_image_grid = gr.Checkbox(label='Generate Image Grid for Each Batch',
                                                          info='(Experimental) This may cause performance problems on some computers and certain internet conditions.',
                                                          value=False)

                        refiner_swap_method = gr.Dropdown(label='Refiner swap method',
                                                          value=flags.refiner_swap_method,
                                                          choices=['joint', 'separate', 'vae'])

                        with gr.Row():
                            switch_sampler = gr.Checkbox(label="Switch", value=False,
                                                         info="Determine whether the #sampler# parameter is required, ignore by default")
                            sampler_name = gr.Dropdown(label='Sampler', choices=flags.sampler_list,
                                                       value=modules.config.default_sampler)
                            scheduler_name = gr.Dropdown(label='Scheduler', choices=flags.scheduler_list,
                                                         value=modules.config.default_scheduler)

                        with gr.Row():
                            adm_scaler_positive = gr.Slider(label='Positive ADM Guidance Scaler', minimum=0.1,
                                                            maximum=3.0,
                                                            step=0.001, value=1.5,
                                                            info='The scaler multiplied to positive ADM (use 1.0 to disable). ')
                            adm_scaler_negative = gr.Slider(label='Negative ADM Guidance Scaler', minimum=0.1,
                                                            maximum=3.0,
                                                            step=0.001, value=0.8,
                                                            info='The scaler multiplied to negative ADM (use 1.0 to disable). ')
                        with gr.Row():
                            adm_scaler_end = gr.Slider(label='ADM Guidance End At Step', minimum=0.0, maximum=1.0,
                                                       step=0.001, value=0.3,
                                                       info='When to end the guidance from positive/negative ADM. ')

                            adaptive_cfg = gr.Slider(label='CFG Mimicking from TSNR', minimum=1.0, maximum=30.0,
                                                     step=0.01,
                                                     value=modules.config.default_cfg_tsnr,
                                                     info='Enabling Fooocus\'s implementation of CFG mimicking for TSNR '
                                                          '(effective when real CFG > mimicked CFG).')
                        with gr.Row():
                            overwrite_step = gr.Slider(label='Forced Overwrite of Sampling Step',
                                                       minimum=-1, maximum=200, step=1,
                                                       value=modules.config.default_overwrite_step,
                                                       info='Set as -1 to disable. For developer debugging.')
                            overwrite_switch = gr.Slider(label='Forced Overwrite of Refiner Switch Step',
                                                         minimum=-1, maximum=200, step=1,
                                                         value=modules.config.default_overwrite_switch,
                                                         info='Set as -1 to disable. For developer debugging.')
                        with gr.Row():
                            overwrite_width = gr.Slider(label='Forced Overwrite of Generating Width',
                                                        minimum=-1, maximum=2048, step=1, value=-1,
                                                        info='Set as -1 to disable. For developer debugging. '
                                                             'Results will be worse for non-standard numbers that SDXL is not trained on.')
                            overwrite_height = gr.Slider(label='Forced Overwrite of Generating Height',
                                                         minimum=-1, maximum=2048, step=1, value=-1,
                                                         info='Set as -1 to disable. For developer debugging. '
                                                              'Results will be worse for non-standard numbers that SDXL is not trained on.')
                        with gr.Row():
                            overwrite_vary_strength = gr.Slider(
                                label='Forced Overwrite of Denoising Strength of "Vary"',
                                minimum=-1, maximum=1.0, step=0.001, value=-1,
                                info='Set as negative number to disable. For developer debugging.')
                            overwrite_upscale_strength = gr.Slider(
                                label='Forced Overwrite of Denoising Strength of "Upscale"',
                                minimum=-1, maximum=1.0, step=0.001, value=-1,
                                info='Set as negative number to disable. For developer debugging.')

                        with gr.Row():
                            disable_intermediate_results = gr.Checkbox(label='Disable Intermediate Results',
                                                                       value=modules.config.default_performance == 'LCM',
                                                                       interactive=modules.config.default_performance != 'LCM',
                                                                       info='Disable intermediate results during generation, only show final gallery. eg LCM')
                            disable_seed_increment = gr.Checkbox(label='Disable seed increment',
                                                                 info='Disable automatic seed increment when image number is > 1.',
                                                                 value=False)

                            read_wildcards_in_order = gr.Checkbox(label="Read wildcards in order", value=False)

                    with gr.Tab(label='Control'):
                        debugging_cn_preprocessor = gr.Checkbox(label='Debug Preprocessors', value=False,
                                                                info='See the results from preprocessors.')
                        skipping_cn_preprocessor = gr.Checkbox(label='Skip Preprocessors', value=False,
                                                               info='Do not preprocess images. (Inputs are already canny/depth/cropped-face/etc.)')

                        mixing_image_prompt_and_vary_upscale = gr.Checkbox(
                            label='Mixing Image Prompt and Vary/Upscale',
                            value=False)
                        mixing_image_prompt_and_inpaint = gr.Checkbox(label='Mixing Image Prompt and Inpaint',
                                                                      value=False)

                        controlnet_softness = gr.Slider(label='Softness of ControlNet', minimum=0.0, maximum=1.0,
                                                        step=0.001, value=0.25,
                                                        info='Similar to the Control Mode in A1111 (use 0.0 to disable). ')

                        with gr.Tab(label='Canny'):
                            with gr.Row():
                                canny_low_threshold = gr.Slider(label='Canny Low Threshold', minimum=1, maximum=255,
                                                                step=1, value=64)
                                canny_high_threshold = gr.Slider(label='Canny High Threshold', minimum=1,
                                                                 maximum=255,
                                                                 step=1, value=128)

                    with gr.Tab(label='Inpaint'):
                        debugging_inpaint_preprocessor = gr.Checkbox(label='Debug Inpaint Preprocessing',
                                                                     value=False)
                        inpaint_disable_initial_latent = gr.Checkbox(label='Disable initial latent in inpaint',
                                                                     value=False)
                        inpaint_engine = gr.Dropdown(label='Inpaint Engine',
                                                     value=modules.config.default_inpaint_engine_version,
                                                     choices=flags.inpaint_engine_versions,
                                                     info='Version of Fooocus inpaint model')
                        inpaint_strength = gr.Slider(label='Inpaint Denoising Strength',
                                                     minimum=0.0, maximum=1.0, step=0.001, value=1.0,
                                                     info='Same as the denoising strength in A1111 inpaint. '
                                                          'Only used in inpaint, not used in outpaint. '
                                                          '(Outpaint always use 1.0)')
                        inpaint_respective_field = gr.Slider(label='Inpaint Respective Field',
                                                             minimum=0.0, maximum=1.0, step=0.001, value=0.618,
                                                             info='The area to inpaint. '
                                                                  'Value 0 is same as "Only Masked" in A1111. '
                                                                  'Value 1 is same as "Whole Image" in A1111. '
                                                                  'Only used in inpaint, not used in outpaint. '
                                                                  '(Outpaint always use 1.0)')
                        inpaint_erode_or_dilate = gr.Slider(label='Mask Erode or Dilate',
                                                            minimum=-64, maximum=64, step=1, value=0,
                                                            info='Positive value will make white area in the mask larger, '
                                                                 'negative value will make white area smaller.'
                                                                 '(default is 0, always process before any mask invert)')
                        inpaint_mask_upload_checkbox = gr.Checkbox(label='Enable Mask Upload', value=False)
                        invert_mask_checkbox = gr.Checkbox(label='Invert Mask', value=False)

                        inpaint_ctrls = [debugging_inpaint_preprocessor, inpaint_disable_initial_latent,
                                         inpaint_engine,
                                         inpaint_strength, inpaint_respective_field,
                                         inpaint_mask_upload_checkbox, invert_mask_checkbox,
                                         inpaint_erode_or_dilate]

                        inpaint_mask_upload_checkbox.change(lambda x: gr.update(visible=x),
                                                            inputs=inpaint_mask_upload_checkbox,
                                                            outputs=inpaint_mask_image, queue=False,
                                                            show_progress=False)

                    with gr.Tab(label='FreeU'):
                        freeu_enabled = gr.Checkbox(label='Enabled', value=False)
                        with gr.Row():
                            freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
                            freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
                        with gr.Row():
                            freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
                            freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)
                        freeu_ctrls = [freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2]


                def dev_mode_checked(r):
                    return gr.update(visible=r)


                dev_mode.change(dev_mode_checked, inputs=[dev_mode], outputs=[dev_tools],
                                queue=False, show_progress=False)

            with gr.Tab(label='Misc'):
                output_format = gr.Radio(label='Output Format',
                                         choices=modules.flags.output_formats,
                                         value=modules.config.default_output_format)
                with gr.Row():
                    save_metadata_to_images = gr.Checkbox(label='Save Metadata to Images',
                                                          value=True,
                                                          info='Adds parameters to generated images allowing manual regeneration.')
                    metadata_scheme = gr.Radio(label='Metadata Scheme', choices=flags.metadata_scheme,
                                               value=modules.config.default_metadata_scheme,
                                               info='Image Prompt parameters are not included. Use png and a1111 for compatibility with Civitai.',
                                               visible=True)

                    save_metadata_to_images.change(lambda x: gr.update(visible=x),
                                                   inputs=[save_metadata_to_images],
                                                   outputs=[metadata_scheme],
                                                   queue=False, show_progress=False)

                    save_metadata_json = gr.Checkbox(label='Save Metadata to JSON',
                                                     value=settings['save_metadata_json'])
                    # save_metadata_image = gr.Checkbox(label='Save Metadata to Image',
                    #                                   value=settings['save_metadata_image'])

                metadata_viewer = gr.JSON(label='Metadata')


            def trigger_describe(mode, img):
                if mode == flags.desc_type_photo:
                    from extras.interrogate import \
                        default_interrogator as default_interrogator_photo
                    return default_interrogator_photo(img), ["Fooocus V2", "Fooocus Enhance",
                                                             "Fooocus Sharp"]
                if mode == flags.desc_type_anime:
                    from extras.wd14tagger import default_interrogator as default_interrogator_anime
                    return default_interrogator_anime(img), ["Fooocus V2", "Fooocus Masterpiece"]
                return mode, ["Fooocus V2"]


            desc_btn.click(trigger_describe, inputs=[desc_method, desc_input_image],
                           outputs=[prompt, style_selections], show_progress=True, queue=True)


        def adjust_refiner_model_config(performance, preset_name, current_refiner):
            preset_config = modules.config.get_config_from_model_preset(preset_name) or {}
            default_refiner = preset_config.get("default_refiner")

            performance_name = str(performance or "").strip()
            performance_key = performance_name.lower()

            if performance_name == "Custom":
                requested_refiner = current_refiner
            else:
                requested_refiner = default_refiner

            if performance_key == constants.TYPE_LIGHTNING.lower():
                model_filter = constants.TYPE_LIGHTNING.lower()
                label = f"{performance_name.title()} model(for SDXL)"
            elif performance_key in {
                constants.TYPE_LCM.lower(),
                constants.TYPE_TURBO.lower(),
                constants.TYPE_HYPER_SD.lower(),
            }:
                model_filter = performance_key
                label = f"{performance_name.title()} model(for SDXL)"
            else:
                choices = ["None"] + list(modules.config.model_filenames)
                value = requested_refiner if requested_refiner in choices else "None"

                return gr.update(
                    label="Refiner (SDXL or SD 1.5)",
                    choices=choices,
                    value=value,
                    show_label=True,
                )

            filtered_models = [
                filename
                for filename in modules.config.model_filenames
                if model_filter in filename.lower()
            ]
            choices = ["None"] + filtered_models
            value = requested_refiner if requested_refiner in choices else "None"

            return gr.update(
                label=label,
                choices=choices,
                value=value,
                show_label=True,
            )


        def reset_model_preset(x):
            results = []
            real_resolution = modules.config.add_ratio(
                modules.config.get_config_from_model_preset(x).get("default_aspect_ratio"))
            printF(name=MasterName.get_master_name(),
                   info="[Parameters] width × height: {}".format(real_resolution)).printf()

            results += [modules.config.get_config_from_model_preset(x).get("default_performance")]
            results += [x]
            results += [real_resolution]
            results += [modules.config.get_config_from_model_preset(x).get("default_prompt")]
            results += [modules.config.get_config_from_model_preset(x).get("default_prompt_negative")]

            results += [modules.config.get_config_from_model_preset(x).get("default_model_type")]
            m = modules.config.get_config_from_model_preset(x).get("default_model")

            if m in modules.config.model_filenames:
                results.append(m)
            else:
                results.append("None")

            loras = modules.config.get_config_from_model_preset(x).get("default_loras")
            for ll in loras:
                if ll[1] in modules.config.lora_filenames:
                    results += [ll[0], ll[1], ll[2]]
                else:
                    results += [False, "None", 1]

            results += [modules.config.get_config_from_model_preset(x).get("default_sample_sharpness")]
            results += [modules.config.get_config_from_model_preset(x).get("default_cfg_scale")]
            results += [modules.config.get_config_from_model_preset(x).get("default_sampler")]
            results += [modules.config.get_config_from_model_preset(x).get("default_scheduler")]
            results += [modules.config.get_config_from_model_preset(x).get("default_refiner_switch")]
            results += [modules.config.get_config_from_model_preset(x).get("default_styles")]
            results += [modules.config.get_config_from_model_preset(x).get("remark")]

            r = modules.config.get_config_from_model_preset(x).get("default_refiner")
            if r in modules.config.model_filenames:
                results += [r]
            else:
                results += ["None"]

            gr.Info(str(x) + ' in effect!')
            return results


        model_presets.change(fn=reset_model_preset, inputs=model_presets,
                             outputs=[performance_selection, model_presets, aspect_ratios_selection, prompt,
                                      negative_prompt, model_type_selector,
                                      base_model] + lora_ctrls + [sharpness, guidance_scale, sampler_name,
                                                                  scheduler_name, refiner_switch, style_selections,
                                                                  model_lora_remark, refiner_model]) \
            .then(fn=adjust_refiner_model_config, inputs=[performance_selection, model_presets, refiner_model],
                  outputs=refiner_model)


        def model_refresh_clicked(*x):
            modules.config.update_all_model_names()
            results = []

            if x[0] not in modules.config.preset_filenames:
                results += [gr.update(
                    choices=['Not Exist!->'] + modules.config.preset_filenames,
                    value="Not Exist!->"
                )]
            else:
                results += [gr.update(choices=modules.config.preset_filenames)]

            selected_model_filenames = modules.config.get_model_filenames(modules.config.modelfile_path,
                                                                          name_filter=x[1])
            selected_model_filenames = sorted(set(selected_model_filenames), key=selected_model_filenames.index)

            results += [gr.update(value=x[1])]

            base_choices = ['None'] + selected_model_filenames
            base_value = x[2] if x[2] in base_choices else "None"
            
            results += [
                gr.update(
                    choices=base_choices,
                    value=base_value
                )
            ]

            # 修正：原代码 `x[3] not in [list, "None"]` 永远为真
            refiner_choices = ['None'] + modules.config.model_filenames
            refiner_value = x[3] if x[3] in refiner_choices else refiner_choices[0]

            results += [
                gr.update(
                    choices=refiner_choices,
                    value=refiner_value
                )
            ]

            y = list(x[4:-2])
            z = [y[nn:nn + 3] for nn in range(0, len(y), 3)]
            for lf in z:
                if lf[1] is not None and "\\" in lf[1]:
                    lf[1] = lf[1].replace('\\\\', '\\')
                if lf[1] is None or lf[1] not in modules.config.lora_filenames:
                    missing_value = "None" if lf[1] is None else "Not Exist!->"
                    available = ['None', 'Not Exist!->'] + modules.config.lora_filenames
                    results += [gr.update(value=False),
                                gr.update(choices=available, value=missing_value),
                                gr.update(value=lf[2])]
                else:
                    results += [gr.update(value=True),
                                gr.update(choices=['None'] + modules.config.lora_filenames, value=lf[1]),
                                gr.update(value=lf[2])]

            if x[-2] not in modules.config.controlnet_lora_canny_filenames:
                results += [gr.update(
                    choices=['None', 'Not Exist!->'] + modules.config.controlnet_lora_canny_filenames,
                    value="Not Exist!->"
                )]
            else:
                results += [gr.update(choices=['None'] + modules.config.controlnet_lora_canny_filenames)]

            if x[-1] not in modules.config.controlnet_lora_depth_filenames:
                results += [gr.update(
                    choices=['None', 'Not Exist!->'] + modules.config.controlnet_lora_depth_filenames,
                    value="Not Exist!->"
                )]
            else:
                results += [gr.update(choices=['None'] + modules.config.controlnet_lora_depth_filenames)]

            gr.Info("All Model Info UPDATED!")
            return results


        model_refresh.click(model_refresh_clicked,
                            [model_presets, model_type_selector, base_model, refiner_model] + lora_ctrls + [canny_model,
                                                                                                            depth_model],
                            [model_presets, model_type_selector, base_model, refiner_model] + lora_ctrls + [canny_model,
                                                                                                            depth_model],
                            queue=False, show_progress=False)


        def forbid_performance_settings(x1):
            # outputs = [
            # sharpness, adm_scaler_end, adm_scaler_positive,adm_scaler_negative, refiner_switch, refiner_model, adaptive_cfg,
            # sampler_name, scheduler_name,
            # refiner_swap_method,
            # cfg
            # ]
            result = []
            if x1 not in ['LCM', 'TURBO', 'Custom', 'Lightning']:
                for rlt in [2.0, 0.3, 1.5, 0.8, 0.8, "None", 7.0]:
                    result += [gr.update(interactive=True, value=rlt)]
                result += [gr.update(interactive=True, value="dpmpp_2m_sde_gpu"),
                           gr.update(interactive=True, value="karras")]
                result += [gr.update(interactive=True)]
                result += [gr.update(interactive=True, value=4.0)]
            elif x1 == 'LCM':
                # refiner_swap_method 不改变数值
                for rlt in [2.0, 0.0, 1.0, 1.0, 1.0, "None", 1.0]:
                    result += [gr.update(interactive=False, value=rlt)]
                result += [gr.update(interactive=True, value="lcm"), gr.update(interactive=True, value="lcm")]
                result += [gr.update(interactive=False)]
                result += [gr.update(interactive=True, value=1.0)]
            elif x1 == 'Lightning':
                # refiner_swap_method 不改变数值
                for rlt in [2.0, 0.0, 1.0, 1.0, 1.0]:
                    result += [gr.update(interactive=False, value=rlt)]
                result += [gr.update(interactive=True, value="None"), gr.update(interactive=False, value="1.0")]
                result += [gr.update(interactive=True, value="euler"),
                           gr.update(interactive=True, value="sgm_uniform")]
                result += [gr.update(interactive=False)]
                result += [gr.update(interactive=True, value=1.0)]
            elif x1 == 'TURBO':
                # refiner_swap_method 不改变数值
                for rlt in [2.0, 0.0, 1.0, 1.0, 1.0, "None", 1.0]:
                    result += [gr.update(interactive=False, value=rlt)]
                result += [gr.update(interactive=True, value="euler_ancestral"),
                           gr.update(interactive=True, value="karras")]
                result += [gr.update(interactive=False)]
                result += [gr.update(interactive=True, value=1.0)]
            elif x1 == 'HYPER_SD':
                # refiner_swap_method 不改变数值
                for rlt in [0.0, 0.0, 1.0, 1.0, 1.0, "None", 1.0]:
                    result += [gr.update(interactive=False, value=rlt)]
                result += [gr.update(interactive=True, value="dpmpp_sde_gpu"),
                           gr.update(interactive=True, value="karras")]
                result += [gr.update(interactive=False)]
                result += [gr.update(interactive=True, value=1.0)]
            elif x1 == 'Custom':
                # refiner_swap_method 不改变数值
                result += [gr.update(interactive=True)] * 11
            return result


        performance_selection.change(fn=forbid_performance_settings,
                                     inputs=performance_selection,
                                     outputs=[
                                         sharpness, adm_scaler_end, adm_scaler_positive,
                                         adm_scaler_negative, refiner_switch, refiner_model, adaptive_cfg,
                                         sampler_name, scheduler_name, refiner_swap_method, guidance_scale
                                     ], queue=False, show_progress=False) \
            .then(fn=adjust_refiner_model_config, inputs=[performance_selection, model_presets, refiner_model],
                  outputs=refiner_model)

        output_format.input(lambda x: gr.update(output_format=x), inputs=output_format)

        image_factory_checkbox.change(lambda x: gr.update(visible=x), image_factory_checkbox, advanced_column,
                                      queue=False, show_progress=False) \
            .then(fn=lambda: None, js='refresh_grid_delayed', queue=False, show_progress=False)


        def img2img_mode_checked(x):
            return gr.update(visible=x), gr.update(visible=x), gr.update(visible=x)


        img2img_mode.change(fn=img2img_mode_checked, inputs=[img2img_mode],
                            outputs=[image_2_image_panel, input_gallery, revision_gallery], queue=False)

        load_input_images_button.upload(fn=load_input_images_handler, inputs=[load_input_images_button],
                                        outputs=[input_gallery, gallery_tabs, image_number])
        load_revision_images_button.upload(fn=load_revision_images_handler, inputs=[load_revision_images_button],
                                           outputs=[revision_mode, revision_gallery, gallery_tabs])
        output_to_input_button.click(output_to_input_handler, inputs=output_gallery,
                                     outputs=[input_gallery, gallery_tabs])
        output_to_revision_button.click(output_to_revision_handler, inputs=output_gallery,
                                        outputs=[revision_mode, revision_gallery, gallery_tabs])


        def verify_enhance_image(enhance_image, img2img):
            if enhance_image and img2img:
                gr.Warning('Image-2-Image: disabled (Enhance Image priority)')
                return gr.update(value=False)
            else:
                return gr.update()


        def verify_input(img2img, canny, depth, gallery_in, gallery_rev, gallery_out):
            if (img2img or canny or depth) and len(gallery_in) == 0:
                if len(gallery_rev) > 0:
                    gr.Info('Image-2-Image / CL: imported revision as input')
                    return gr.update(), gr.update(), gr.update(), list(map(lambda x: x['name'], gallery_rev[:1]))
                elif len(gallery_out) > 0:
                    gr.Info('Image-2-Image / CL: imported output as input')
                    return gr.update(), gr.update(), gr.update(), list(map(lambda x: x['name'], gallery_out[:1]))
                else:
                    gr.Warning('Image-2-Image / CL: disabled (no images available)')
                    return gr.update(value=False), gr.update(value=False), gr.update(value=False), gr.update()
            else:
                return gr.update(), gr.update(), gr.update(), gr.update()


        def inpaint_mode_change(mode):
            assert mode in modules.flags.inpaint_options

            # inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
            # inpaint_disable_initial_latent, inpaint_engine,
            # inpaint_strength, inpaint_respective_field

            if mode == modules.flags.inpaint_option_detail:
                return [
                    gr.update(visible=True), gr.update(visible=False, value=[]),
                    gr.Dataset.update(visible=True, samples=modules.config.example_inpaint_prompts),
                    False, 'None', 0.5, 0.0
                ]

            if mode == modules.flags.inpaint_option_modify:
                return [
                    gr.update(visible=True), gr.update(visible=False, value=[]),
                    gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
                    True, modules.config.default_inpaint_engine_version, 1.0, 0.0
                ]

            return [
                gr.update(visible=False, value=''), gr.update(visible=True),
                gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
                False, modules.config.default_inpaint_engine_version, 1.0, 0.618
            ]


        inpaint_mode.input(inpaint_mode_change, inputs=inpaint_mode, outputs=[
            inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
            inpaint_disable_initial_latent, inpaint_engine,
            inpaint_strength, inpaint_respective_field
        ], show_progress=False, queue=False)

        # ctrls
        ctrls = [currentTask, generate_image_grid]
        ctrls += [
            prompt, negative_prompt, style_selections,
            performance_selection, aspect_ratios_selection, image_number, image_seed,
            sharpness, switch_sampler, sampler_name, scheduler_name, fixed_steps, custom_steps, custom_switch,
            guidance_scale
        ]

        ctrls += [base_model, refiner_model, base_clip_skip, refiner_clip_skip, refiner_switch] + lora_ctrls
        ctrls += [image_factory_checkbox, current_tab]
        ctrls += [uov_method, uov_input_image]
        ctrls += [outpaint_selections, outpaint_expansion_ratio, inpaint_input_image, inpaint_additional_prompt,
                  inpaint_mask_image]
        ctrls += [disable_preview, disable_intermediate_results, disable_seed_increment]
        ctrls += [adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg]
        ctrls += [overwrite_step, overwrite_switch, overwrite_width, overwrite_height, overwrite_vary_strength]
        ctrls += [overwrite_upscale_strength, mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint]
        ctrls += [debugging_cn_preprocessor, skipping_cn_preprocessor, canny_low_threshold, canny_high_threshold]
        ctrls += [refiner_swap_method, controlnet_softness]
        ctrls += freeu_ctrls
        ctrls += inpaint_ctrls

        # conflict with fun in metadata to image, remove --> save_metadata_image
        # ctrls += [save_metadata_json, save_metadata_image] + img2img_ctrls + [same_seed_for_all, output_format]
        ctrls += [save_metadata_json] + img2img_ctrls + [same_seed_for_all, output_format]
        ctrls += canny_ctrls + depth_ctrls
        ctrls += ip_ctrls
        ctrls += [model_type_selector]
        ctrls += [nsfw_filter]

        load_prompt_button.upload(fn=load_prompt_handler, inputs=[load_prompt_button] + ctrls + [seed_random],
                                  outputs=ctrls + [seed_random])
        load_last_prompt_button.click(fn=load_last_prompt_handler, inputs=ctrls + [seed_random],
                                      outputs=ctrls + [seed_random])

        if not adapter.args_manager.args.disable_metadata:
            ctrls += [save_metadata_to_images, metadata_scheme]

        nums_ctrls = len(ctrls)
        printF(name=MasterName.get_master_name(), info="WebUI Server init ctrls: {}".format(nums_ctrls)).printf()

        groups = []
        names_dict = {}
        num = 4
        string = ""
        for inx, val in enumerate(ctrls):
            name = modules.util.get_var_name(val)[0]
            names_dict[name] = val
            if num <= 4:
                string += "|{0:<2}| - {1:<30} - {2:<15}".format(inx, name, str(val.value))
                num -= 1
                if inx > nums_ctrls - 4 and (num == int(nums_ctrls) // 4):
                    groups.append(string)
            if num == 0 and string != "":
                groups.append(string)
                string = ""
                num = 4

        for kk in groups:
            printF(name=MasterName.get_master_name(), info="{}".format(kk)).printf()

        # foooocus code
        # def refresh_files_clicked():
        #     modules.config.update_all_model_names()
        #     results = [gr.update(choices=modules.config.model_filenames)]
        #     results += [gr.update(choices=['None'] + modules.config.model_filenames)]
        #     if not adapter.args_manager.args.disable_preset_selection:
        #         results += [gr.update(choices=modules.config.available_presets)]
        #     for i in range(modules.config.default_max_lora_number):
        #         results += [gr.update(interactive=True),
        #                     gr.update(choices=['None'] + modules.config.lora_filenames), gr.update()]
        #     return results
        #
        #
        # refresh_files_output = [base_model, refiner_model]
        # if not adapter.args_manager.args.disable_preset_selection:
        #     refresh_files_output += [preset_selection]
        # refresh_files.click(refresh_files_clicked, [], refresh_files_output + lora_ctrls,
        #                     queue=False, show_progress=False)
        # fooocus code

        state_is_generating = gr.State(False)


        def parse_meta(raw_prompt_txt, is_generating):
            loaded_json = None
            if is_json(raw_prompt_txt):
                loaded_json = json.loads(raw_prompt_txt)

            if loaded_json is None:
                if is_generating:
                    return gr.update(), gr.update(), gr.update()
                else:
                    return gr.update(), gr.update(visible=True), gr.update(visible=False)

            return json.dumps(loaded_json), gr.update(visible=False), gr.update(visible=True)


        # prompt.input(parse_meta, inputs=[prompt, state_is_generating],
        #              outputs=[prompt, generate_button, load_parameter_button], queue=False, show_progress=False)
        #
        load_data_outputs = [image_number, prompt, negative_prompt, style_selections,
                             performance_selection, overwrite_step, overwrite_switch, aspect_ratios_selection,
                             overwrite_width, overwrite_height, guidance_scale, sharpness, adm_scaler_positive,
                             adm_scaler_negative, adm_scaler_end, refiner_swap_method, adaptive_cfg, base_model,
                             refiner_model, refiner_switch, sampler_name, scheduler_name, seed_random, image_seed,
                             generate_button] + freeu_ctrls + lora_ctrls


        #
        # load_parameter_button.click(modules.meta_parser.load_parameter_button_click,
        #                             inputs=[prompt, state_is_generating], outputs=load_data_outputs, queue=False,
        #                             show_progress=False)

        # fooocus code begin
        # if not adapter.args_manager.args.disable_preset_selection:
        #     def preset_selection_change(preset, is_generating):
        #         preset_content = modules.config.try_get_preset_content(preset) if preset != 'initial' else {}
        #         preset_prepared = modules.meta_parser.parse_meta_from_preset(preset_content)
        #
        #         default_model = preset_prepared.get('base_model')
        #         previous_default_models = preset_prepared.get('previous_default_models', [])
        #         checkpoint_downloads = preset_prepared.get('checkpoint_downloads', {})
        #         embeddings_downloads = preset_prepared.get('embeddings_downloads', {})
        #         lora_downloads = preset_prepared.get('lora_downloads', {})
        #
        #         preset_prepared['base_model'], preset_prepared['lora_downloads'] = PreCheck.download_models(
        #             default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads)
        #
        #         if 'prompt' in preset_prepared and preset_prepared.get('prompt') == '':
        #             del preset_prepared['prompt']
        #
        #         return modules.meta_parser.load_parameter_button_click(json.dumps(preset_prepared), is_generating)
        #
        #
        #     preset_selection.change(preset_selection_change, inputs=[preset_selection, state_is_generating],
        #                             outputs=load_data_outputs, queue=False, show_progress=True) \
        #         .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False,
        #               show_progress=False)
        # fooocus code end

        def trigger_metadata_import(filepath, state_is_generating):
            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(filepath)
            if parameters is None:
                printF(name=MasterName.get_master_name(), info="[ERROR] Could not find metadata in the image!").printf()
                parsed_parameters = {}
            else:
                metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
                parsed_parameters = metadata_parser.parse_json(parameters)

            return modules.meta_parser.load_parameter_button_click(parsed_parameters, state_is_generating)


        metadata_import_button.click(trigger_metadata_import, inputs=[metadata_input_image, state_is_generating],
                                     outputs=load_data_outputs, queue=False, show_progress=True)
        #     .then(style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False,
        #           show_progress=False)

        generate_button.click(lambda: (
            gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True),
            gr.update(visible=False, interactive=False), [], True),
                              outputs=[stop_button, skip_button, generate_button, output_gallery,
                                       state_is_generating]) \
            .then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed) \
            .then(fn=verify_enhance_image, inputs=[image_factory_checkbox, img2img_mode], outputs=[img2img_mode]) \
            .then(fn=verify_input,
                  inputs=[img2img_mode, control_lora_canny, control_lora_depth, input_gallery, revision_gallery,
                          output_gallery],
                  outputs=[img2img_mode, control_lora_canny, control_lora_depth, input_gallery]) \
            .then(fn=verify_revision, inputs=[revision_mode, input_gallery, revision_gallery, output_gallery],
                  outputs=[revision_mode, revision_gallery]) \
            .then(fn=get_task, inputs=ctrls + [input_gallery, revision_gallery, keep_input_names], outputs=currentTask) \
            .then(fn=generate_clicked, inputs=currentTask,
                  outputs=[progress_html, progress_window, remain_images_progress, gallery_holder, output_gallery,
                           progress_gallery, finish_image_viewer,
                           metadata_viewer, gallery_tabs]) \
            .then(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False),
                           gr.update(visible=False, interactive=False), False),
                  outputs=[generate_button, stop_button, skip_button, state_is_generating]) \
            .then(fn=update_history_link, outputs=history_link) \
            .then(fn=lambda: None, js='playNotification').then(fn=lambda: None, js='refresh_grid_delayed')

        # for notification_file in ['notification.ogg', 'notification.mp3']:
        #     if os.path.exists(notification_file):
        #         gr.Audio(interactive=False, value=notification_file, elem_id='audio_notification', visible=False)
        #         break
                

# 在 launch 之前添加类型检查和转换
path_outputs = modules.config.path_outputs
auth_filename = constants.AUTH_FILENAME

if not isinstance(path_outputs, str):
    print(f"Warning: path_outputs is {type(path_outputs)}, expected str. Setting allowed_paths=None")
    allowed_paths = None
else:
    allowed_paths = [path_outputs]

if not isinstance(auth_filename, str):
    print(f"Warning: AUTH_FILENAME is {type(auth_filename)}, expected str. Setting blocked_paths=None")
    blocked_paths = None
else:
    blocked_paths = [auth_filename]

# 处理 server_name 避免 localhost 不可访问
server_name = adapter.args_manager.args.listen
if not server_name or server_name == "0.0.0.0":
    # 0.0.0.0 可能导致 localhost 检查失败，改为 127.0.0.1 或保持但设置 share=True
    # 最简单：如果 share=False 且 server_name 为 0.0.0.0，强制 share=True
    if not adapter.args_manager.args.share and server_name == "0.0.0.0":
        print("Warning: share=False with server_name='0.0.0.0' may cause localhost error. Setting share=True.")
        share = True
    else:
        share = adapter.args_manager.args.share
else:
    share = adapter.args_manager.args.share
async_gradio_app = shared.gradio_root
async_gradio_app.launch(
    inbrowser=adapter.args_manager.args.in_browser,
    server_name=server_name or "127.0.0.1",
    server_port=adapter.args_manager.args.port or 7860,
    share=share,
    auth=check_auth if (share or server_name) and auth_enabled else None,
    allowed_paths=allowed_paths,
    blocked_paths=blocked_paths
)