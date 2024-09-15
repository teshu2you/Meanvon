import ssl
from util.printf import printF, MasterName
from config.webuiConfig import *

ssl._create_default_https_context = ssl._create_unverified_context
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
from extensions.sadtalker.src.gradio_demo import SadTalker
from resources import *
from resources.musicgen_mel import modellist_musicgen_mel, initiate_stop_musicgen_mel, music_musicgen_mel
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
        if len(task.yields) > 0:
            flag, product = task.yields.pop(0)
            # [progress_html, progress_window, progress_gallery, gallery
            # progress_gallery   gallery=output_gallery , both in gallery_tabs, gallery_tabs in gallery_holder
            # progress_html, progress_window, remain_images_progress, gallery_holder, output_gallery, progress_gallery, finish_image_viewer, metadata_viewer, gallery_tabs
            if flag == 'preview':
                # help bad internet connection by skipping duplicated preview
                if len(task.yields) > 0:  # if we have the next item
                    if task.yields[0][0] == 'preview':  # if the next item is also a preview
                        # print('Skipped one preview for better internet connection.')
                        continue
                percentage, title, image, img_pp, img_rr = product
                yield gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                    gr.update(visible=True, value=image) if image is not None else gr.update(), \
                    gr.update(visible=True,
                              value="No." + str(img_pp) + " processing...           |           " + str(
                                  img_rr) + "  image(s) pending!"), \
                    gr.update(visible=False), \
                    gr.update(), \
                    gr.update(), \
                    gr.update(open=True), \
                    gr.update(), \
                    gr.update()
            if flag == 'metadatas':
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(
                    value=product), gr.update(selected=GALLERY_ID_OUTPUT)
            if flag == 'results':
                yield gr.update(visible=True), \
                    gr.update(visible=True), \
                    gr.update(visible=True, value="Partially done"), \
                    gr.update(visible=False), \
                    gr.update(), \
                    gr.update(visible=True, value=product) if product is not None else gr.update(visible=False), \
                    gr.update(open=True), \
                    gr.update(), \
                    gr.update(selected=GALLERY_ID_OUTPUT)
            if flag == 'finish':
                yield gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=True, value="All done"), \
                    gr.update(visible=True), \
                    gr.update(value=product), \
                    gr.update(value=product), \
                    gr.update(open=False), \
                    gr.update(), \
                    gr.update(selected=GALLERY_ID_OUTPUT)
                finished = True

                # delete Fooocus temp images, only keep gradio temp images
                if adapter.args_manager.args.disable_image_log:
                    for filepath in product:
                        if isinstance(filepath, str) and os.path.exists(filepath):
                            os.remove(filepath)

    execution_time = time.perf_counter() - execution_start_time
    print(f'Total time: {execution_time:.2f} seconds')
    return


def metadata_to_ctrls(metadata, ctrls):
    # important webui parameters!
    if not isinstance(metadata, Mapping):
        return ctrls

    if 'prompt' in metadata:
        ctrls[2] = metadata.get('prompt')
    if 'negative_prompt' in metadata:
        ctrls[3] = metadata.get('negative_prompt')
    if 'styles' in metadata:
        ctrls[4] = ast.literal_eval(metadata.get('styles'))
    elif 'style' in metadata:
        ctrls[4] = migrate_style_from_v1(metadata.get('style'))
    if 'performance' in metadata:
        ctrls[5] = metadata.get('performance')
    if 'width' in metadata and 'height' in metadata:
        ctrls[6] = get_resolution_new_string(metadata.get('width'), metadata.get('height'))
    elif 'resolution' in metadata:
        ctrls[6] = metadata.get('resolution')
    # image_number
    if 'image_number' in metadata:
        ctrls[7] = metadata.get('image_number')
    if 'seed' in metadata:
        ctrls[8] = metadata.get('seed')
    if 'sharpness' in metadata:
        ctrls[9] = metadata.get('sharpness')
    # ctrls[10] switch_sampler skip
    if 'sampler_name' in metadata:
        ctrls[11] = metadata.get('sampler_name')
    elif 'sampler' in metadata:
        ctrls[11] = metadata.get('sampler')
    if 'scheduler' in metadata:
        ctrls[12] = metadata.get('scheduler')
    if 'steps' in metadata:
        ctrls[13] = int(metadata.get('steps'))
        ctrls[14] = int(metadata.get('steps'))
    if 'switch' in metadata:
        ctrls[15] = round(metadata.get('switch') / metadata.get('steps'), 2)
        # if ctrls[12] != round(constants.SWITCH_SPEED / constants.STEPS_SPEED, 2):
        #     ctrls[3] = 'Custom'
    if 'cfg' in metadata:
        ctrls[16] = metadata.get('cfg')

    if 'guidance_scale' in metadata:
        ctrls[16] = metadata.get('guidance_scale')

    if 'base_model' in metadata:
        _tmp = metadata.get('base_model')
        if ".safetensors" not in _tmp and _tmp not in ['None', 'none', 'Not Exist!->']:
            ctrls[17] = _tmp + ".safetensors"
        else:
            ctrls[17] = _tmp
    elif 'base_model_name' in metadata:
        _tmp = metadata.get('base_model_name')
        if ".safetensors" not in _tmp and _tmp not in ['None', 'none', 'Not Exist!->']:
            ctrls[17] = _tmp + ".safetensors"
        else:
            ctrls[17] = _tmp
    if 'refiner_model' in metadata:
        _tmp = metadata.get('refiner_model')
        if ".safetensors" not in _tmp and _tmp not in ['None', 'none', 'Not Exist!->']:
            ctrls[18] = _tmp + ".safetensors"
        else:
            ctrls[18] = _tmp
    elif 'refiner_model_name' in metadata:
        _tmp = metadata.get('refiner_model_name')
        if ".safetensors" not in _tmp and _tmp not in ['None', 'none', 'Not Exist!->']:
            ctrls[18] = _tmp + ".safetensors"
        else:
            ctrls[18] = _tmp
    if 'base_clip_skip' in metadata:
        ctrls[19] = metadata.get('base_clip_skip')
    if 'refiner_clip_skip' in metadata:
        ctrls[20] = metadata.get('refiner_clip_skip')
    if 'refiner_switch' in metadata:
        ctrls[21] = metadata.get('refiner_switch')

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
            ctrls[lora_begin_idx + 2] = kv[1].strip()
        else:
            ctrls[lora_begin_idx] = False
            ctrls[lora_begin_idx + 1] = "None"
            ctrls[lora_begin_idx + 2] = 1
        lora_begin_idx += 3

    # if there are more than 5 loras, ignore them. (can not be seen on webui)
    #  'lora_combined_1': 'Primary\\SDXL_LORA_(Movie Still)_JuggerCineXL2.safetensors : 0.42', 'lora_combined_2': 'Primary\\SDXL_LORA_控制_add-detail-xl增加细节.safetensors : 0.69', 'lora_combined_3': 'Primary\\SDXL_LORA_艺术_more_art-full_v1.safetensors : 0.76',

    # seed_random 无需设置 , not all parameters should be set here, just use above.
    if 'model_type_selector' in metadata:
        ctrls[122] = metadata.get('model_type_selector')
    if 'seed_random' in metadata:
        ctrls[123] = not metadata.get('seed_random')

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
settings = default_settings
reload_javascript()

title = f'MeanVon {main_version}'

if isinstance(adapter.args_manager.args.preset, str):
    title += ' ' + adapter.args_manager.args.preset


def change_model_type_llamacpp(model_llamacpp):
    try:
        test_model = model_list_llamacpp[model_llamacpp]
    except KeyError as ke:
        test_model = None
    if (test_model != None):
        return prompt_template_llamacpp.update(
            value=model_list_llamacpp[model_llamacpp][1]), system_template_llamacpp.update(
            value=model_list_llamacpp[model_llamacpp][2]), quantization_llamacpp.update(value="")
    else:
        return prompt_template_llamacpp.update(value="{prompt}"), system_template_llamacpp.update(
            value=""), quantization_llamacpp.update(value="")


def change_prompt_template_llamacpp(prompt_template):
    return prompt_template_llamacpp.update(
        value=prompt_template_list_llamacpp[prompt_template][0]), system_template_llamacpp.update(
        value=prompt_template_list_llamacpp[prompt_template][1])


## Functions specific to llamacpp
def show_download_llamacpp():
    return btn_download_file_llamacpp.update(visible=False), download_file_llamacpp.update(visible=True)


def hide_download_llamacpp():
    return btn_download_file_llamacpp.update(visible=True), download_file_llamacpp.update(visible=False)


def change_model_type_llamacpp(model_llamacpp):
    try:
        test_model = model_list_llamacpp[model_llamacpp]
    except KeyError as ke:
        test_model = None
    if (test_model != None):
        return prompt_template_llamacpp.update(
            value=model_list_llamacpp[model_llamacpp][1]), system_template_llamacpp.update(
            value=model_list_llamacpp[model_llamacpp][2]), quantization_llamacpp.update(value="")
    else:
        return prompt_template_llamacpp.update(value="{prompt}"), system_template_llamacpp.update(
            value=""), quantization_llamacpp.update(value="")


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


with (gr.Blocks(
        title=title,
        theme=gr.themes.Soft(primary_hue=gr.themes.colors.rose,
                             secondary_hue=gr.themes.colors.lime,
                             neutral_hue=gr.themes.colors.indigo
                             ).set(
            body_background_fill="linear-gradient(white 1px, transparent 0), linear-gradient(90deg, white 1px, transparent 0)"),
        css=modules.html.css) as shared.gradio_root):
    currentTask = gr.State(task_manager.AsyncTask(args=[]))
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


            ## Functions specific to AnimateLCM
            def read_ini_animatediff_lcm(module):
                # model_animatediff_lcm.value = readcfg_animatediff_lcm[0]
                # adapter_animatediff_lcm.value = readcfg_animatediff_lcm[1]
                # lora_animatediff_lcm.value = readcfg_animatediff_lcm[2]
                # num_inference_step_animatediff_lcm.value = readcfg_animatediff_lcm[3]
                # sampler_animatediff_lcm.value = readcfg_animatediff_lcm[4]
                # guidance_scale_animatediff_lcm.value = readcfg_animatediff_lcm[5]
                # seed_animatediff_lcm.value = readcfg_animatediff_lcm[6]
                # num_frames_animatediff_lcm.value = readcfg_animatediff_lcm[7]
                # width_animatediff_lcm.value = readcfg_animatediff_lcm[8]
                # height_animatediff_lcm.value = readcfg_animatediff_lcm[9]
                # num_videos_per_prompt_animatediff_lcm.value = readcfg_animatediff_lcm[10]
                # num_prompt_animatediff_lcm.value = readcfg_animatediff_lcm[11]
                # use_gfpgan_animatediff_lcm.value = readcfg_animatediff_lcm[12]
                # tkme_animatediff_lcm.value = readcfg_animatediff_lcm[13]

                content = read_ini(module)
                return str(content[0]), str(content[1]), str(content[2]), int(content[3]), str(content[4]), float(
                    content[5]), int(content[6]), int(
                    content[7]), int(content[8]), int(content[9]), int(content[10]), int(content[11]), bool(
                    int(content[12])), float(content[13])


            def read_ini_animatediff_lightning(module):
                content = read_ini(module)
                return str(content[0]), str(content[1]), int(content[2]), str(content[3]), float(
                    content[4]), int(content[5]), int(
                    content[6]), int(content[7]), int(content[8]), int(content[9]), int(content[10]), bool(
                    int(content[11])), float(content[12])


            ## Functions specific to MusicGen Melody
            def read_ini_musicgen_mel(module):
                content = read_ini(module)
                return str(content[0]), int(content[1]), float(content[2]), int(content[3]), bool(
                    int(content[4])), float(content[5]), int(content[6]), int(content[7])


            def change_source_type_musicgen_mel(source_type_musicgen_mel):
                if source_type_musicgen_mel == "audio":
                    return source_audio_musicgen_mel.update(source="upload")
                elif source_type_musicgen_mel == "micro":
                    return source_audio_musicgen_mel.update(source="microphone")


            ## Functions specific to Bark
            def read_ini_bark(module):
                content = read_ini(module)
                return str(content[0]), str(content[1])


            def read_ini_llava(module):
                content = read_ini(module)
                return str(content[0]), int(content[1]), int(content[2]), bool(int(content[3])), int(
                    content[4]), float(
                    content[5]), float(content[6]), float(content[7]), int(content[8]), str(content[9])


            def show_download_llava():
                return btn_download_file_llava.update(visible=False), download_file_llava.update(visible=True)


            def hide_download_llava():
                return btn_download_file_llava.update(visible=True), download_file_llava.update(visible=False)


            def change_model_type_animatediff_lcm(model_animatediff_lcm):
                if (model_animatediff_lcm == "stabilityai/sdxl-turbo"):
                    return sampler_animatediff_lcm.update(
                        value="LCM"), width_animatediff_lcm.update(), height_animatediff_lcm.update(), num_inference_step_animatediff_lcm.update(
                        value=2), guidance_scale_animatediff_lcm.update(
                        value=0.0), negative_prompt_animatediff_lcm.update(interactive=False)
                elif ("XL" in model_animatediff_lcm.upper()) or (model_animatediff_lcm == "segmind/SSD-1B") or (
                        model_animatediff_lcm == "dataautogpt3/OpenDalleV1.1"):
                    return sampler_animatediff_lcm.update(
                        value="LCM"), width_animatediff_lcm.update(), height_animatediff_lcm.update(), num_inference_step_animatediff_lcm.update(
                        value=10), guidance_scale_animatediff_lcm.update(
                        value=7.5), negative_prompt_animatediff_lcm.update(interactive=True)
                elif (model_animatediff_lcm == "segmind/Segmind-Vega"):
                    return sampler_animatediff_lcm.update(
                        value="LCM"), width_animatediff_lcm.update(), height_animatediff_lcm.update(), num_inference_step_animatediff_lcm.update(
                        value=10), guidance_scale_animatediff_lcm.update(
                        value=9.0), negative_prompt_animatediff_lcm.update(interactive=True)
                else:
                    return sampler_animatediff_lcm.update(
                        value="LCM"), width_animatediff_lcm.update(), height_animatediff_lcm.update(), num_inference_step_animatediff_lcm.update(
                        value=10), guidance_scale_animatediff_lcm.update(), negative_prompt_animatediff_lcm.update(
                        interactive=True)


            def change_output_type_animatediff_lcm(output_type_animatediff_lcm):
                if output_type_animatediff_lcm == "mp4":
                    return out_animatediff_lcm.update(visible=True), gif_out_animatediff_lcm.update(
                        visible=False), btn_animatediff_lcm.update(visible=True), btn_animatediff_lcm_gif.update(
                        visible=False)
                elif output_type_animatediff_lcm == "gif":
                    return out_animatediff_lcm.update(visible=False), gif_out_animatediff_lcm.update(
                        visible=True), btn_animatediff_lcm.update(visible=False), btn_animatediff_lcm_gif.update(
                        visible=True)


            def change_model_type_animatediff_lightning(model_animatediff_lightning):
                if (model_animatediff_lightning == "stabilityai/sdxl-turbo"):
                    return sampler_animatediff_lightning.update(
                        value="Euler"), width_animatediff_lightning.update(), height_animatediff_lightning.update(), num_inference_step_animatediff_lightning.update(
                        value=2), guidance_scale_animatediff_lightning.update(
                        value=0.0), negative_prompt_animatediff_lightning.update(interactive=False)
                elif ("XL" in model_animatediff_lightning.upper()) or (
                        model_animatediff_lightning == "segmind/SSD-1B") or (
                        model_animatediff_lightning == "dataautogpt3/OpenDalleV1.1"):
                    return sampler_animatediff_lightning.update(
                        value="Euler"), width_animatediff_lightning.update(), height_animatediff_lightning.update(), num_inference_step_animatediff_lightning.update(
                        value=10), guidance_scale_animatediff_lightning.update(
                        value=7.5), negative_prompt_animatediff_lightning.update(interactive=True)
                elif (model_animatediff_lightning == "segmind/Segmind-Vega"):
                    return sampler_animatediff_lightning.update(
                        value="Euler"), width_animatediff_lightning.update(), height_animatediff_lightning.update(), num_inference_step_animatediff_lightning.update(
                        value=10), guidance_scale_animatediff_lightning.update(
                        value=9.0), negative_prompt_animatediff_lightning.update(interactive=True)
                else:
                    return sampler_animatediff_lightning.update(
                        value="Euler"), width_animatediff_lightning.update(), height_animatediff_lightning.update(), num_inference_step_animatediff_lightning.update(
                        value=10), guidance_scale_animatediff_lightning.update(), negative_prompt_animatediff_lightning.update(
                        interactive=True)


            def change_output_type_txt2prompt(output_type_txt2prompt):
                if output_type_txt2prompt == "ChatGPT":
                    return model_txt2prompt.update(value=model_list_txt2prompt[1]), max_tokens_txt2prompt.update(
                        value=128)
                elif output_type_txt2prompt == "SD":
                    return model_txt2prompt.update(value=model_list_txt2prompt[0]), max_tokens_txt2prompt.update(
                        value=70)


            with gr.Row(visible=False) as text_input_panel:
                with gr.Tabs():
                    with gr.TabItem(f"Chatbot Llama-cpp (gguf) 📝", id=11) as tab_llamacpp:
                        with gr.Accordion(f"About", open=False):
                            with gr.Box():
                                gr.HTML(
                                    f"""
                                                    <h1 style='text-align: left;'>{about_infos}</h1>
                                                    <b>{about_module}</b>{tab_llamacpp}</br>
                                                    <b>{about_function}</b>{tab_llamacpp_about_desc} <a href='https://github.com/abetlen/llama-cpp-python' target='_blank'>llama-cpp-python</a></br>
                                                    <b>{about_inputs}</b>{about_input_text}</br>
                                                    <b>{about_outputs}</b>{about_output_text}</br>
                                                    <b>{about_modelpage}</b>
                                                    <a href='https://hf-mirror.com/zhouzr/Llama3-8B-Chinese-Chat-GGUF' target='_blank'>zhouzr/Llama3-8B-Chinese-Chat-GGUF</a>, 
                                                    <a href='https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct-GGUF' target='_blank'>NousResearch/Meta-Llama-3-8B-Instruct-GGUF</a>, 
                                                    <a href='https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF' target='_blank'>Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF</a>, 
                                                    <a href='https://huggingface.co/bartowski/gemma-2-9b-it-GGUF' target='_blank'>bartowski/gemma-2-9b-it-GGUF</a>, 
                                                    <a href='https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF' target='_blank'>bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF</a>, 
                                                    <a href='https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF' target='_blank'>NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF</a>, 
                                                    <a href='https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf' target='_blank'>microsoft/Phi-3-mini-4k-instruct-gguf</a>, 
                                                    <a href='https://huggingface.co/bartowski/openchat-3.6-8b-20240522-GGUF' target='_blank'>bartowski/openchat-3.6-8b-20240522-GGUF</a>, 
                                                    <a href='https://huggingface.co/LoneStriker/Starling-LM-7B-beta-GGUF' target='_blank'>LoneStriker/Starling-LM-7B-beta-GGUF</a>, 
                                                    <a href='https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B-GGUF' target='_blank'>NousResearch/Hermes-2-Pro-Mistral-7B-GGUF</a>, 
                                                    <a href='https://huggingface.co/Lewdiculous/Kunoichi-DPO-v2-7B-GGUF-Imatrix' target='_blank'>Lewdiculous/Kunoichi-DPO-v2-7B-GGUF-Imatrix</a>, 
                                                    <a href='https://huggingface.co/dranger003/MambaHermes-3B-GGUF' target='_blank'>dranger003/MambaHermes-3B-GGUF</a>, 
                                                    <a href='https://huggingface.co/bartowski/gemma-1.1-7b-it-GGUF' target='_blank'>bartowski/gemma-1.1-7b-it-GGUF</a>, 
                                                    <a href='https://huggingface.co/bartowski/gemma-1.1-2b-it-GGUF' target='_blank'>bartowski/gemma-1.1-2b-it-GGUF</a>, 
                                                    <a href='https://huggingface.co/mlabonne/AlphaMonarch-7B-GGUF' target='_blank'>mlabonne/AlphaMonarch-7B-GGUF</a>, 
                                                    <a href='https://huggingface.co/mlabonne/NeuralBeagle14-7B-GGUF' target='_blank'>mlabonne/NeuralBeagle14-7B-GGUF</a>, 
                                                    <a href='https://huggingface.co/TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF' target='_blank'>TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF</a>, 
                                                    <a href='https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF' target='_blank'>TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF</a>, 
                                                    <a href='https://huggingface.co/TheBloke/phi-2-GGUF' target='_blank'>TheBloke/phi-2-GGUF</a>, 
                                                    <a href='https://huggingface.co/TheBloke/Mixtral_7Bx2_MoE-GGUF' target='_blank'>TheBloke/Mixtral_7Bx2_MoE-GGUF</a>, 
                                                    <a href='https://huggingface.co/TheBloke/mixtralnt-4x7b-test-GGUF' target='_blank'>TheBloke/mixtralnt-4x7b-test-GGUF</a>, 
                                                    <a href='https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF' target='_blank'>bartowski/Mistral-7B-Instruct-v0.3-GGUF</a>, 
                                                    <a href='https://huggingface.co/TheBloke/MetaMath-Cybertron-Starling-GGUF' target='_blank'>TheBloke/MetaMath-Cybertron-Starling-GGUF</a>, 
                                                    <a href='https://huggingface.co/TheBloke/una-cybertron-7B-v2-GGUF' target='_blank'>TheBloke/una-cybertron-7B-v2-GGUF</a>, 
                                                    <a href='https://huggingface.co/TheBloke/Starling-LM-7B-alpha-GGUF' target='_blank'>TheBloke/Starling-LM-7B-alpha-GGUF</a>, 
                                                    <a href='https://huggingface.co/TheBloke/neural-chat-7B-v3-2-GGUF' target='_blank'>TheBloke/neural-chat-7B-v3-2-GGUF</a>, 
                                                    <a href='https://huggingface.co/TheBloke/CollectiveCognition-v1.1-Mistral-7B-GGUF' target='_blank'>TheBloke/CollectiveCognition-v1.1-Mistral-7B-GGUF</a>, 
                                                    <a href='https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF' target='_blank'>TheBloke/zephyr-7B-beta-GGUF</a>, 
                                                    <a href='https://huggingface.co/TheBloke/Yarn-Mistral-7B-128k-GGUF' target='_blank'>TheBloke/Yarn-Mistral-7B-128k-GGUF</a>, 
                                                    <a href='https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF' target='_blank'>TheBloke/CodeLlama-13B-Instruct-GGUF</a></br>
                                                    """
                                    #                                <a href='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF' target='_blank'>TheBloke/Mistral-7B-Instruct-v0.2-GGUF</a>,
                                )
                            with gr.Box():
                                gr.HTML(
                                    f"""
                                                    <h1 style='text-align: left;'>{about_help}</h1>
                                                    <div style='text-align: justified'>
                                                    <b>{about_usage}</b></br>
                                                    {tab_llamacpp_about_instruct}
                                                    </br>
                                                    <b>{about_models}</b></br>
                                                    - {tab_llamacpp_about_models_inst1}</br>
                                                    - {tab_llamacpp_about_models_inst2}
                                                    </div>
                                                    """
                                )
                        with gr.Accordion(factory_settings, open=False):
                            with gr.Row():
                                with gr.Column():
                                    model_llamacpp = gr.Dropdown(choices=list(model_list_llamacpp.keys()),
                                                                 value=list(model_list_llamacpp.keys())[0],
                                                                 label=model_label, allow_custom_value=True,
                                                                 info=tab_llamacpp_model_info)
                                with gr.Column():
                                    quantization_llamacpp = gr.Textbox(value="",
                                                                       label=tab_llamacpp_quantization_label,
                                                                       info=tab_llamacpp_quantization_info)
                                with gr.Column():
                                    max_tokens_llamacpp = gr.Slider(0, 524288, step=16, value=1024,
                                                                    label=maxtoken_label,
                                                                    info=maxtoken_info)
                                with gr.Column():
                                    seed_llamacpp = gr.Slider(0, 10000000000, step=1, value=1337,
                                                              label=seed_label, info=seed_info)
                            with gr.Row():
                                with gr.Column():
                                    stream_llamacpp = gr.Checkbox(value=False, label=stream_label,
                                                                  info=stream_info, interactive=False)
                                with gr.Column():
                                    n_ctx_llamacpp = gr.Slider(0, 131072, step=128, value=8192,
                                                               label=ctx_label, info=ctx_info)
                                with gr.Column():
                                    repeat_penalty_llamacpp = gr.Slider(0.0, 10.0, step=0.1, value=1.1,
                                                                        label=penalty_label,
                                                                        info=penalty_info)
                            with gr.Row():
                                with gr.Column():
                                    temperature_llamacpp = gr.Slider(0.0, 10.0, step=0.1, value=0.8,
                                                                     label=temperature_label,
                                                                     info=temperature_info)
                                with gr.Column():
                                    top_p_llamacpp = gr.Slider(0.0, 10.0, step=0.05, value=0.95,
                                                               label=top_p_label,
                                                               info=top_p_info)
                                with gr.Column():
                                    top_k_llamacpp = gr.Slider(0, 500, step=1, value=40, label=top_k_label,
                                                               info=top_k_info)
                            with gr.Row():
                                with gr.Column():
                                    force_prompt_template_llamacpp = gr.Dropdown(
                                        choices=list(prompt_template_list_llamacpp.keys()),
                                        value=list(prompt_template_list_llamacpp.keys())[0],
                                        label=tab_llamacpp_force_prompt_label,
                                        info=tab_llamacpp_force_prompt_info)
                                with gr.Column():
                                    gr.Number(visible=False)
                                with gr.Column():
                                    gr.Number(visible=False)
                            with gr.Row():
                                with gr.Column():
                                    prompt_template_llamacpp = gr.Textbox(label=prompt_template_label,
                                                                          value=
                                                                          model_list_llamacpp[model_llamacpp.value][1],
                                                                          lines=4, max_lines=4, show_copy_button=True,
                                                                          info=prompt_template_info)
                            with gr.Row():
                                with gr.Column():
                                    system_template_llamacpp = gr.Textbox(label=system_template_label,
                                                                          value=
                                                                          model_list_llamacpp[model_llamacpp.value][2],
                                                                          lines=4, max_lines=4, show_copy_button=True,
                                                                          info=system_template_info)
                                    model_llamacpp.change(fn=change_model_type_llamacpp, inputs=model_llamacpp,
                                                          outputs=[prompt_template_llamacpp, system_template_llamacpp,
                                                                   quantization_llamacpp])
                                    force_prompt_template_llamacpp.change(fn=change_prompt_template_llamacpp,
                                                                          inputs=force_prompt_template_llamacpp,
                                                                          outputs=[prompt_template_llamacpp,
                                                                                   system_template_llamacpp])
                            with gr.Row():
                                with gr.Column():
                                    save_ini_btn_llamacpp = gr.Button(f"{save_settings} 💾")
                                with gr.Column():
                                    module_name_llamacpp = gr.Textbox(value="llamacpp", visible=False,
                                                                      interactive=False)
                                    del_ini_btn_llamacpp = gr.Button(f"{delete_settings} 🗑️",
                                                                     interactive=True if test_ini_exist(
                                                                         module_name_llamacpp.value) else False)
                                    save_ini_btn_llamacpp.click(
                                        fn=write_ini_llamacpp,
                                        inputs=[
                                            module_name_llamacpp,
                                            model_llamacpp,
                                            quantization_llamacpp,
                                            max_tokens_llamacpp,
                                            seed_llamacpp,
                                            stream_llamacpp,
                                            n_ctx_llamacpp,
                                            repeat_penalty_llamacpp,
                                            temperature_llamacpp,
                                            top_p_llamacpp,
                                            top_k_llamacpp,
                                            force_prompt_template_llamacpp,
                                            prompt_template_llamacpp,
                                            system_template_llamacpp,
                                        ]
                                    )
                                    save_ini_btn_llamacpp.click(fn=lambda: gr.Info(save_settings_msg))
                                    save_ini_btn_llamacpp.click(
                                        fn=lambda: del_ini_btn_llamacpp.update(interactive=True),
                                        outputs=del_ini_btn_llamacpp)
                                    del_ini_btn_llamacpp.click(fn=lambda: del_ini(module_name_llamacpp.value))
                                    del_ini_btn_llamacpp.click(fn=lambda: gr.Info(delete_settings_msg))
                                    del_ini_btn_llamacpp.click(
                                        fn=lambda: del_ini_btn_llamacpp.update(interactive=False),
                                        outputs=del_ini_btn_llamacpp)
                            if test_ini_exist(module_name_llamacpp.value):
                                with open(f".ini/{module_name_llamacpp.value}.ini", "r", encoding="utf-8") as fichier:
                                    exec(fichier.read())
                        with gr.Row():
                            history_llamacpp = gr.Chatbot(
                                label=chatbot_history,
                                height=400,
                                autoscroll=True,
                                show_copy_button=True,
                                interactive=True,
                                bubble_full_width=False,
                                avatar_images=("./background/robot.jpg", "./background/me.jpg"),
                            )
                            last_reply_llamacpp = gr.Textbox(value="", visible=False)
                        with gr.Row():
                            prompt_llamacpp = gr.Textbox(label=chatbot_prompt_label, lines=1, max_lines=3,
                                                         show_copy_button=True,
                                                         placeholder=chatbot_prompt_placeholder,
                                                         autofocus=True)
                            hidden_prompt_llamacpp = gr.Textbox(value="", visible=False)
                            last_reply_llamacpp.change(fn=lambda x: x, inputs=hidden_prompt_llamacpp,
                                                       outputs=prompt_llamacpp)
                        with gr.Row():
                            with gr.Column():
                                btn_llamacpp = gr.Button(f"{generate} 🚀", variant="primary")
                            with gr.Column():
                                btn_llamacpp_continue = gr.Button(f"{factory_continue} ➕")
                            with gr.Column():
                                btn_llamacpp_clear_output = gr.ClearButton(components=[history_llamacpp],
                                                                           value=f"{clear_outputs} 🧹")
                            with gr.Column():
                                btn_download_file_llamacpp = gr.ClearButton(value=f"{download_chat} 💾",
                                                                            visible=True)
                                download_file_llamacpp = gr.File(label=f"{download_chat}",
                                                                 value=blankfile_common, height=30, interactive=False,
                                                                 visible=False)
                                download_file_llamacpp_hidden = gr.Textbox(value=blankfile_common, interactive=False,
                                                                           visible=False)
                                btn_download_file_llamacpp.click(fn=show_download_llamacpp,
                                                                 outputs=[btn_download_file_llamacpp,
                                                                          download_file_llamacpp])
                                download_file_llamacpp_hidden.change(fn=lambda x: x,
                                                                     inputs=download_file_llamacpp_hidden,
                                                                     outputs=download_file_llamacpp)
                            btn_llamacpp.click(
                                fn=text_llamacpp,
                                inputs=[
                                    model_llamacpp,
                                    quantization_llamacpp,
                                    max_tokens_llamacpp,
                                    seed_llamacpp,
                                    stream_llamacpp,
                                    n_ctx_llamacpp,
                                    repeat_penalty_llamacpp,
                                    temperature_llamacpp,
                                    top_p_llamacpp,
                                    top_k_llamacpp,
                                    prompt_llamacpp,
                                    history_llamacpp,
                                    prompt_template_llamacpp,
                                    system_template_llamacpp,
                                ],
                                outputs=[
                                    history_llamacpp,
                                    last_reply_llamacpp,
                                    download_file_llamacpp_hidden,
                                ],
                                show_progress="full",
                            )
                            btn_llamacpp.click(fn=hide_download_llamacpp,
                                               outputs=[btn_download_file_llamacpp, download_file_llamacpp])
                            prompt_llamacpp.submit(
                                fn=text_llamacpp,
                                inputs=[
                                    model_llamacpp,
                                    quantization_llamacpp,
                                    max_tokens_llamacpp,
                                    seed_llamacpp,
                                    stream_llamacpp,
                                    n_ctx_llamacpp,
                                    repeat_penalty_llamacpp,
                                    temperature_llamacpp,
                                    top_p_llamacpp,
                                    top_k_llamacpp,
                                    prompt_llamacpp,
                                    history_llamacpp,
                                    prompt_template_llamacpp,
                                    system_template_llamacpp,
                                ],
                                outputs=[
                                    history_llamacpp,
                                    last_reply_llamacpp,
                                    download_file_llamacpp_hidden,
                                ],
                                show_progress="full",
                            )
                            prompt_llamacpp.submit(fn=hide_download_llamacpp,
                                                   outputs=[btn_download_file_llamacpp, download_file_llamacpp])
                            btn_llamacpp_continue.click(
                                fn=text_llamacpp_continue,
                                inputs=[
                                    model_llamacpp,
                                    quantization_llamacpp,
                                    max_tokens_llamacpp,
                                    seed_llamacpp,
                                    stream_llamacpp,
                                    n_ctx_llamacpp,
                                    repeat_penalty_llamacpp,
                                    temperature_llamacpp,
                                    top_p_llamacpp,
                                    top_k_llamacpp,
                                    history_llamacpp,
                                ],
                                outputs=[
                                    history_llamacpp,
                                    last_reply_llamacpp,
                                    download_file_llamacpp_hidden,
                                ],
                                show_progress="full",
                            )
                            btn_llamacpp_continue.click(fn=hide_download_llamacpp,
                                                        outputs=[btn_download_file_llamacpp, download_file_llamacpp])

                    with gr.TabItem("Llava 1.5 (gguf) 👁️", id=12) as tab_llava:
                        with gr.Accordion("About", open=False):
                            with gr.Box():
                                gr.HTML(
                                    """
                                    <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                    <b>Module : </b>Llava 1.5 (gguf)</br>
                                    <b>Function : </b>Interrogate a chatbot about an input image using <a href='https://github.com/abetlen/llama-cpp-python' target='_blank'>llama-cpp-python</a>, <a href='https://llava-vl.github.io/' target='_blank'>Llava 1.5</a> and <a href='https://github.com/SkunkworksAI/BakLLaVA' target='_blank'>BakLLaVA</a></br>
                                    <b>Input(s) : </b>Input image, Input text</br>
                                    <b>Output(s) : </b>Output text</br>
                                    <b>HF models pages : </b>
                                    <a href='https://huggingface.co/mys/ggml_bakllava-1' target='_blank'>mys/ggml_bakllava-1</a>, 
                                    <a href='https://huggingface.co/mys/ggml_llava-v1.5-7b' target='_blank'>mys/ggml_llava-v1.5-7b</a>, 
                                    <a href='https://huggingface.co/mys/ggml_llava-v1.5-13b' target='_blank'>mys/ggml_llava-v1.5-13b</a>
                               </br>
                                    """
                                )
                            with gr.Box():
                                gr.HTML(
                                    """
                                    <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                    <div style='text-align: justified'>
                                    <b>Usage :</b></br>
                                    - Upload or import an <b>Input image</b></br>
                                    - Type your request in the <b>Input</b> textbox field</br>
                                    - (optional) modify settings to use another model, change context size or modify maximum number of tokens generated.</br>
                                    - Click the <b>Generate</b> button to generate a response to your input, using the chatbot history to keep a context.</br>
                                    - Click the <b>Continue</b> button to complete the last reply.
                                    </br>
                                    <b>Models :</b></br>
                                    - You could place llama-cpp compatible .gguf models in the directory ./models/llava. Restart to see them in the models list.
                                    </div>
                                    """
                                )
                        with gr.Accordion("Settings", open=False):
                            with gr.Row():
                                with gr.Column():
                                    model_llava = gr.Dropdown(choices=model_list_llava, value=model_list_llava[0],
                                                              label="Model",
                                                              info="Choose model to use for inference")
                                with gr.Column():
                                    max_tokens_llava = gr.Slider(0, 131072, step=16, value=512, label="Max tokens",
                                                                 info="Maximum number of tokens to generate")
                                with gr.Column():
                                    seed_llava = gr.Slider(0, 10000000000, step=1, value=1337,
                                                           label="Seed(0 for random)",
                                                           info="Seed to use for generation.")
                            with gr.Row():
                                with gr.Column():
                                    stream_llava = gr.Checkbox(value=False, label="Stream", info="Stream results",
                                                               interactive=False)
                                with gr.Column():
                                    n_ctx_llava = gr.Slider(0, 131072, step=128, value=8192, label="n_ctx",
                                                            info="Maximum context size")
                                with gr.Column():
                                    repeat_penalty_llava = gr.Slider(0.0, 10.0, step=0.1, value=1.1,
                                                                     label="Repeat penalty",
                                                                     info="The penalty to apply to repeated tokens")
                            with gr.Row():
                                with gr.Column():
                                    temperature_llava = gr.Slider(0.0, 10.0, step=0.1, value=0.8,
                                                                  label="Temperature",
                                                                  info="Temperature to use for sampling")
                                with gr.Column():
                                    top_p_llava = gr.Slider(0.0, 10.0, step=0.05, value=0.95, label="top_p",
                                                            info="The top-p value to use for sampling")
                                with gr.Column():
                                    top_k_llava = gr.Slider(0, 500, step=1, value=40, label="top_k",
                                                            info="The top-k value to use for sampling")
                            with gr.Row():
                                with gr.Column():
                                    prompt_template_llava = gr.Textbox(label="Prompt template", value="{prompt}",
                                                                       lines=4, max_lines=4,
                                                                       info="Place your custom prompt template here. Keep the {prompt} tag, that will be replaced by your prompt.")
                            with gr.Row():
                                with gr.Column():
                                    save_ini_btn_llava = gr.Button("Save custom defaults settings 💾")
                                with gr.Column():
                                    module_name_llava = gr.Textbox(value="llava", visible=False, interactive=False)
                                    del_ini_btn_llava = gr.Button("Delete custom defaults settings 🗑️",
                                                                  interactive=True if test_cfg_exist(
                                                                      module_name_llava.value) else False)
                                    save_ini_btn_llava.click(
                                        fn=write_ini,
                                        inputs=[
                                            module_name_llava,
                                            model_llava,
                                            max_tokens_llava,
                                            seed_llava,
                                            stream_llava,
                                            n_ctx_llava,
                                            repeat_penalty_llava,
                                            temperature_llava,
                                            top_p_llava,
                                            top_k_llava,
                                            prompt_template_llava,
                                        ]
                                    )
                                    save_ini_btn_llava.click(fn=lambda: gr.Info('Settings saved'))
                                    save_ini_btn_llava.click(fn=lambda: del_ini_btn_llava.update(interactive=True),
                                                             outputs=del_ini_btn_llava)
                                    del_ini_btn_llava.click(fn=lambda: del_ini(module_name_llava.value))
                                    del_ini_btn_llava.click(fn=lambda: gr.Info('Settings deleted'))
                                    del_ini_btn_llava.click(fn=lambda: del_ini_btn_llava.update(interactive=False),
                                                            outputs=del_ini_btn_llava)
                            if test_cfg_exist(module_name_llava.value):
                                readcfg_llava = read_ini_llava(module_name_llava.value)
                                model_llava.value = readcfg_llava[0]
                                max_tokens_llava.value = readcfg_llava[1]
                                seed_llava.value = readcfg_llava[2]
                                stream_llava.value = readcfg_llava[3]
                                n_ctx_llava.value = readcfg_llava[4]
                                repeat_penalty_llava.value = readcfg_llava[5]
                                temperature_llava.value = readcfg_llava[6]
                                top_p_llava.value = readcfg_llava[7]
                                top_k_llava.value = readcfg_llava[8]
                                prompt_template_llava.value = readcfg_llava[9]
                        with gr.Row():
                            with gr.Column(scale=1):
                                img_llava = gr.Image(label="Input image", type="filepath", height=400)
                            with gr.Column(scale=3):
                                history_llava = gr.Chatbot(
                                    label="Chatbot history",
                                    height=400,
                                    autoscroll=True,
                                    show_copy_button=True,
                                    interactive=True,
                                    bubble_full_width=False,
                                    avatar_images=("./background/avatar.png", "./background/matt.png"),
                                )
                                last_reply_llava = gr.Textbox(value="", visible=False)
                        with gr.Row():
                            prompt_llava = gr.Textbox(label="Input", lines=1, max_lines=3,
                                                      placeholder="Type your request here ...", autofocus=True)
                            hidden_prompt_llava = gr.Textbox(value="", visible=False)
                        with gr.Row():
                            btn_llava = gr.Button("Generate 🚀", variant="primary")
                            btn_llava_clear_input = gr.ClearButton(components=[img_llava, prompt_llava],
                                                                   value="Clear inputs 🧹")
                            btn_llava_continue = gr.Button("Continue ➕", visible=False)
                            btn_llava_clear_output = gr.ClearButton(components=[history_llava],
                                                                    value="Clear outputs 🧹")
                            btn_download_file_llava = gr.ClearButton(value="Download full conversation 💾",
                                                                     visible=True)
                            download_file_llava = gr.File(label="Download full conversation",
                                                          value=blankfile_common, height=30, interactive=False,
                                                          visible=False)
                            download_file_llava_hidden = gr.Textbox(value=blankfile_common, interactive=False,
                                                                    visible=False)
                            btn_download_file_llava.click(fn=show_download_llava,
                                                          outputs=[btn_download_file_llava, download_file_llava])
                            download_file_llava_hidden.change(fn=lambda x: x, inputs=download_file_llava_hidden,
                                                              outputs=download_file_llava)
                            btn_llava.click(
                                fn=text_llava,
                                inputs=[
                                    model_llava,
                                    max_tokens_llava,
                                    seed_llava,
                                    stream_llava,
                                    n_ctx_llava,
                                    repeat_penalty_llava,
                                    temperature_llava,
                                    top_p_llava,
                                    top_k_llava,
                                    img_llava,
                                    prompt_llava,
                                    history_llava,
                                    prompt_template_llava,
                                ],
                                outputs=[
                                    history_llava,
                                    last_reply_llava,
                                    download_file_llava_hidden,
                                ],
                                show_progress="full",
                            )
                            btn_llava.click(fn=hide_download_llava,
                                            outputs=[btn_download_file_llava, download_file_llava])
                            prompt_llava.submit(
                                fn=text_llava,
                                inputs=[
                                    model_llava,
                                    max_tokens_llava,
                                    seed_llava,
                                    stream_llava,
                                    n_ctx_llava,
                                    repeat_penalty_llava,
                                    temperature_llava,
                                    top_p_llava,
                                    top_k_llava,
                                    img_llava,
                                    prompt_llava,
                                    history_llava,
                                    prompt_template_llava,
                                ],
                                outputs=[
                                    history_llava,
                                    last_reply_llava,
                                    download_file_llava_hidden,
                                ],
                                show_progress="full",
                            )
                            prompt_llava.submit(fn=hide_download_llava,
                                                outputs=[btn_download_file_llava, download_file_llava])
                            btn_llava_continue.click(
                                fn=text_llava_continue,
                                inputs=[
                                    model_llava,
                                    max_tokens_llava,
                                    seed_llava,
                                    stream_llava,
                                    n_ctx_llava,
                                    repeat_penalty_llava,
                                    temperature_llava,
                                    top_p_llava,
                                    top_k_llava,
                                    img_llava,
                                    history_llava,
                                ],
                                outputs=[
                                    history_llava,
                                    last_reply_llava,
                                    download_file_llava_hidden,
                                ],
                                show_progress="full",
                            )
                            btn_llava_continue.click(fn=hide_download_llava,
                                                     outputs=[btn_download_file_llava, download_file_llava])
                            btn_llava.click(fn=lambda x: x, inputs=hidden_prompt_llava, outputs=prompt_llava)
                            prompt_llava.submit(fn=lambda x: x, inputs=hidden_prompt_llava, outputs=prompt_llava)

                    with gr.TabItem("nllb translation 👥", id=15) as tab_nllb:
                        with gr.Accordion("About", open=False):
                            with gr.Box():
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
                            with gr.Box():
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
                                    save_ini_btn_nllb.click(fn=lambda: del_ini_btn_nllb.update(interactive=True),
                                                            outputs=del_ini_btn_nllb)
                                    del_ini_btn_nllb.click(fn=lambda: del_ini(module_name_nllb.value))
                                    del_ini_btn_nllb.click(fn=lambda: gr.Info('Settings deleted'))
                                    del_ini_btn_nllb.click(fn=lambda: del_ini_btn_nllb.update(interactive=False),
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
                                    out_nllb = gr.Textbox(label="Output text", lines=9, max_lines=9,
                                                          show_copy_button=True, interactive=False)
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

                    if ram_size() >= 16:
                        titletab_txt2prompt = "Prompt generator 📝"
                    else:
                        titletab_txt2prompt = "Prompt generator ⛔"

                    with gr.TabItem(titletab_txt2prompt, id=16) as tab_txt2prompt:
                        with gr.Accordion("About", open=False):
                            with gr.Box(css=".gradio-container {background-color: red}"):
                                gr.HTML(
                                    """
                                    <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                    <b>Module : </b>Prompt generator</br>
                                    <b>Function : </b>Create complex prompt from a simple instruction.</br>
                                    <b>Input(s) : </b>Prompt</br>
                                    <b>Output(s) : </b>Enhanced output prompt</br>
                                    <b>HF model page : </b>
                                    <a href='https://huggingface.co/PulsarAI/prompt-generator' target='_blank'>PulsarAI/prompt-generator</a>, 
                                    <a href='https://huggingface.co/RamAnanth1/distilgpt2-sd-prompts' target='_blank'>RamAnanth1/distilgpt2-sd-prompts</a>, 
                                    </br>
                                    """
                                )
                            with gr.Box():
                                gr.HTML(
                                    """
                                    <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                    <div style='text-align: justified'>
                                    <b>Usage :</b></br>
                                    - Define a <b>prompt</b></br>
                                    - Choose the type of output to produce : ChatGPT will produce a persona for the chatbot from your input, SD will generate a prompt usable for image and video modules</br>
                                    - Click the <b>Generate</b> button</br>
                                    - After generation, output is displayed in the <b>Output text</b> field. Send them to the desired module (chatbot or media modules).
                                    </div>
                                    """
                                )
                        with gr.Accordion("Settings", open=True):
                            with gr.Row():
                                with gr.Column():
                                    model_txt2prompt = gr.Dropdown(choices=model_list_txt2prompt,
                                                                   value=model_list_txt2prompt[0], label="Model",
                                                                   info="Choose model to use for inference")
                                with gr.Column():
                                    max_tokens_txt2prompt = gr.Slider(0, 2048, step=1, value=128,
                                                                      label="Max tokens",
                                                                      info="Maximum number of tokens in output")
                                with gr.Column():
                                    repetition_penalty_txt2prompt = gr.Slider(0.0, 10.0, step=0.01, value=1.05,
                                                                              label="Repetition penalty",
                                                                              info="The penalty to apply to repeated tokens")
                            with gr.Row():
                                with gr.Column():
                                    seed_txt2prompt = gr.Slider(0, 4294967295, step=1, value=0,
                                                                label="Seed(0 for random)",
                                                                info="Seed to use for generation. Permit reproducibility")
                                with gr.Column():
                                    num_prompt_txt2prompt = gr.Slider(1, 64, step=1, value=1, label="Batch size",
                                                                      info="Number of prompts to generate")
                            with gr.Row():
                                with gr.Column():
                                    save_ini_btn_txt2prompt = gr.Button("Save custom defaults settings 💾")
                                with gr.Column():
                                    module_name_txt2prompt = gr.Textbox(value="txt2prompt", visible=False,
                                                                        interactive=False)
                                    del_ini_btn_txt2prompt = gr.Button("Delete custom defaults settings 🗑️",
                                                                       interactive=True if test_cfg_exist(
                                                                           module_name_txt2prompt.value) else False)
                                    save_ini_btn_txt2prompt.click(
                                        fn=write_ini,
                                        inputs=[
                                            module_name_txt2prompt,
                                            model_txt2prompt,
                                            max_tokens_txt2prompt,
                                            repetition_penalty_txt2prompt,
                                            seed_txt2prompt,
                                            num_prompt_txt2prompt,
                                        ]
                                    )
                                    save_ini_btn_txt2prompt.click(fn=lambda: gr.Info('Settings saved'))
                                    save_ini_btn_txt2prompt.click(
                                        fn=lambda: del_ini_btn_txt2prompt.update(interactive=True),
                                        outputs=del_ini_btn_txt2prompt)
                                    del_ini_btn_txt2prompt.click(fn=lambda: del_ini(module_name_txt2prompt.value))
                                    del_ini_btn_txt2prompt.click(fn=lambda: gr.Info('Settings deleted'))
                                    del_ini_btn_txt2prompt.click(
                                        fn=lambda: del_ini_btn_txt2prompt.update(interactive=False),
                                        outputs=del_ini_btn_txt2prompt)
                            if test_cfg_exist(module_name_txt2prompt.value):
                                readcfg_txt2prompt = read_ini_txt2prompt(module_name_txt2prompt.value)
                                model_txt2prompt.value = readcfg_txt2prompt[0]
                                max_tokens_txt2prompt.value = readcfg_txt2prompt[1]
                                repetition_penalty_txt2prompt.value = readcfg_txt2prompt[2]
                                seed_txt2prompt.value = readcfg_txt2prompt[3]
                                num_prompt_txt2prompt.value = readcfg_txt2prompt[4]
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    prompt_txt2prompt = gr.Textbox(label="Prompt", lines=9, max_lines=9,
                                                                   placeholder="a doctor")
                                with gr.Row():
                                    output_type_txt2prompt = gr.Radio(choices=["ChatGPT", "SD"], value="SD",
                                                                      label="Output type",
                                                                      info="Choose type of prompt to generate")
                                    output_type_txt2prompt.change(fn=change_output_type_txt2prompt,
                                                                  inputs=output_type_txt2prompt,
                                                                  outputs=[model_txt2prompt, max_tokens_txt2prompt])
                            with gr.Column():
                                with gr.Row():
                                    out_txt2prompt = gr.Textbox(label="Output prompt", lines=16, max_lines=16,
                                                                show_copy_button=True, interactive=False)
                        with gr.Row():
                            with gr.Column():
                                btn_txt2prompt = gr.Button("Generate 🚀", variant="primary")
                            with gr.Column():
                                btn_txt2prompt_clear_input = gr.ClearButton(components=[prompt_txt2prompt],
                                                                            value="Clear inputs 🧹")
                            with gr.Column():
                                btn_txt2prompt_clear_output = gr.ClearButton(components=[out_txt2prompt],
                                                                             value="Clear outputs 🧹")
                            btn_txt2prompt.click(
                                fn=text_txt2prompt,
                                inputs=[
                                    model_txt2prompt,
                                    max_tokens_txt2prompt,
                                    repetition_penalty_txt2prompt,
                                    seed_txt2prompt,
                                    num_prompt_txt2prompt,
                                    prompt_txt2prompt,
                                    output_type_txt2prompt,
                                ],
                                outputs=out_txt2prompt,
                                show_progress="full",
                            )

            with gr.Row(elem_classes='advanced_check_row'):
                image_factory_checkbox = gr.Checkbox(label='Image-Factory', value=False, container=True,
                                                     info="| text-to-img | img-to-img |",
                                                     elem_classes='min_check')
                # image_factory_advanced_checkbox = gr.Checkbox(label='Configuration', value=modules.config.default_image_factory_advanced_checkbox, container=True, elem_classes='min_check')
            with gr.Row(visible=False) as image_input_panel:
                with gr.Tabs():
                    with gr.TabItem(label='Image 2 Image') as uov_tab:
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
                            output_to_input_button = gr.Button(label='Output to Input', value='Output to Input',
                                                               elem_classes='type_small_row', min_width=0)
                            output_to_revision_button = gr.Button(label='Output to Revision',
                                                                  value='Output to Revision',
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


                    with gr.TabItem(label='Upscale or Variation') as uov_tab:
                        with gr.Row():
                            with gr.Column():
                                uov_input_image = grh.Image(label='Drag above image to here', source='upload',
                                                            type='numpy')
                            with gr.Column():
                                uov_method = gr.Radio(label='Upscale or Variation:', choices=flags.uov_list,
                                                      value=flags.disabled)
                                gr.HTML(
                                    '<a href="https://github.com/lllyasviel/Fooocus/discussions/390" target="_blank">\U0001F4D4 Document</a>')
                    with gr.TabItem(label='ControlNet') as ip_tab:
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

                                        ip_type.change(lambda x: flags.default_parameters[x], inputs=[ip_type],
                                                       outputs=[ip_stop, ip_weight], queue=False,
                                                       show_progress=False)
                                    ip_ad_cols.append(ad_col)


                        def ip_advance_checked(x):
                            return [gr.update(visible=x)] * len(ip_ad_cols) + \
                                [flags.default_ip] * len(ip_types) + \
                                [flags.default_parameters[flags.default_ip][0]] * len(ip_stops) + \
                                [flags.default_parameters[flags.default_ip][1]] * len(ip_weights)


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

            switch_js = "(x) => {if(x){viewer_to_bottom(100);viewer_to_bottom(500);}else{viewer_to_top();} return x;}"
            down_js = "() => {viewer_to_bottom();}"


            def is_model_imported(model_name):
                # print(f"sys.modules: {sys.modules}")
                return model_name in sys.modules.keys()


            with gr.Row(elem_classes='advanced_check_row'):
                video_factory_checkbox = gr.Checkbox(label='Video-Factory', value=False, container=True,
                                                     info="| text-to-vid | img-to-vid |",
                                                     elem_classes='min_check')
                # video_factory_advanced_checkbox = gr.Checkbox(label='Configuration', value=modules.config.default_video_factory_advanced_checkbox, container=True, elem_classes='min_check')

            with gr.Row(visible=False) as video_input_panel:
                # from modules.model_setting import num_frames, num_steps
                from modules.config import svd_config

                version = svd_config.get("version")
                default_fps = infer_args.get('fps', 1)

                with gr.Tabs():

                    if ram_size() >= 16:
                        titletab_tab_svd = "Stable Video Diffusion 📼"
                    else:
                        titletab_tab_svd = "Stable Video Diffusion ⛔"
                    with gr.TabItem(titletab_tab_svd, id=151) as tab_svd:
                        with gr.Blocks(title='Stable Video Diffusion WebUI', css='css/style.css') as demo:
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
                                        seed = gr.Text(value="random", label="seed (integer or 'random')", )
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

                            examples = [["sdxl_styles_samples/Fooocus V2.png"]]
                            inputs = [image, model_load_flag, resize_image, n_frames, n_steps, seed, decoding_t,
                                      fps_id,
                                      motion_bucket_id,
                                      cond_aug]
                            outputs = [video_out]

                            free_cuda_mem()
                            free_cuda_cache()
                            from modules.model import infer

                            btn.click(infer, inputs=inputs, outputs=outputs)
                            gr.Examples(examples=examples, inputs=inputs, outputs=outputs, fn=infer)
                    if ram_size() >= 16:
                        titletab_tab_sad_talker = "SadTalker 📼"
                    else:
                        titletab_tab_sad_talker = "SadTalker ⛔"
                    with gr.TabItem(titletab_tab_sad_talker, id=152) as tab_sad_talker:
                        gr.Markdown("<div align='center'> <h3> 😭 SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023) </span> </h3> \
                                                    <a style='font-size:11px;' href='https://arxiv.org/abs/2211.12194'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                                                    <a style='font-size:11px;' href='https://sadtalker.github.io'>Homepage</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                                                     <a style='font-size:11px;' href='https://github.com/Winfredy/SadTalker'> Github </div>")

                        with gr.Row():
                            source_image = gr.Image(label="Upload image", source="upload", type="filepath",
                                                    elem_id="img2img_image").style(width=512)

                            driven_audio = gr.Audio(label="Upload audio OR TTS", source="upload", type="filepath")

                            if sys.platform != 'win32':
                                from extensions.sadtalker.src.utils.text2speech import TTSTalker

                                tts_talker = TTSTalker()
                                with gr.Column(variant='panel'):
                                    input_text = gr.Textbox(label="Generating audio from text", lines=5,
                                                            placeholder="please enter some text here, we genreate the audio from text using @Coqui.ai TTS.")
                                    tts = gr.Button('Generate audio', elem_id="sadtalker_audio_generate",
                                                    variant='primary')
                                    tts.click(fn=tts_talker.test, inputs=[input_text], outputs=[driven_audio])

                        with gr.Row():
                            with gr.Column(variant='panel'):
                                gr.Markdown(
                                    "Need help? please visit our [best practice page](https://github.com/OpenTalker/SadTalker/blob/main/docs/best_practice.md) for more detials")
                                # width = gr.Slider(minimum=64, elem_id="img2img_width", maximum=2048, step=8, label="Manually Crop Width", value=512) # img2img_width
                                # height = gr.Slider(minimum=64, elem_id="img2img_height", maximum=2048, step=8, label="Manually Crop Height", value=512) # img2img_width
                                result_dir = gr.Textbox(label="Video Save Path", lines=1, max_lines=2,
                                                        value=modules.config.path_videos_sadtalker_outputs,
                                                        info="video output path")
                                pose_style = gr.Slider(minimum=0, maximum=46, step=1, label="Pose style",
                                                       value=0)  #
                                size_of_image = gr.Radio([256, 512], value=256, label='Face Model Resolution',
                                                         info="use 256/512 model?")  #
                                preprocess_type = gr.Radio(['crop', 'resize', 'full', 'extcrop', 'extfull'],
                                                           value='crop', label='Preprocess',
                                                           info="How to handle input image?")
                                is_still_mode = gr.Checkbox(
                                    label="Still Mode (fewer head motion, works with preprocess `full`)")
                                batch_size = gr.Slider(label="Batch Size in generation", step=1, maximum=10,
                                                       value=2)
                                enhancer = gr.Checkbox(label="GFPGAN as Face enhancer")
                                submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')

                            with gr.Column(variant='panel'):
                                # with gr.Blocks() as sad_talker_progress:
                                #     sad_talker_progress_bar = gr.Textbox(label="progress", visible=True)

                                with gr.Tabs(elem_id="sadtalker_genearted"):
                                    gen_video = gr.Video(label="Generated video", format="mp4").style(width=256)

                        # def show_sad_talker_progress(progress=gr.Progress(track_tqdm=True)):
                        #     progress(0, desc="Starting")
                        #     time.sleep(1)
                        #     progress(0.05)
                        #     for k in progress.tqdm(range(60), desc="fighting"):
                        #         time.sleep(1)
                        #     return gr.update(visible=False)

                        # fn = show_sad_talker_progress, outputs = [sad_talker_progress_bar]) \
                        #         .then(

                        sad_talker = SadTalker(checkpoint_path=modules.config.path_sadtalker_checkpoint,
                                               config_path=modules.config.path_sadtalker_config,
                                               lazy_load=True)
                        submit.click(fn=sad_talker.test,
                                     inputs=[source_image,
                                             driven_audio,
                                             preprocess_type,
                                             is_still_mode,
                                             enhancer,
                                             batch_size,
                                             size_of_image,
                                             pose_style,
                                             result_dir
                                             ],
                                     outputs=[gen_video])
                    if ram_size() >= 16:
                        titletab_tab_animatediff_lcm = "AnimateLCM 📼"
                    else:
                        titletab_tab_animatediff_lcm = "AnimateLCM ⛔"
                    with gr.TabItem(titletab_tab_animatediff_lcm, id=43) as tab_animatediff_lcm:
                        with gr.Accordion("About", open=False):
                            with gr.Box():
                                gr.HTML(
                                    """
                                    <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                    <b>Module : </b>AnimateLCM</br>
                                    <b>Function : </b>Generate video from a prompt and a negative prompt using <a href='https://animatelcm.github.io/' target='_blank'>AnimateLCM</a> with <a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a> Models</br>
                                    <b>Input(s) : </b>Prompt, negative prompt</br>
                                    <b>Output(s) : </b>Video</br>
                                    <b>HF model page : </b>
                                    <a href='https://huggingface.co/emilianJR/epiCRealism' target='_blank'>emilianJR/epiCRealism</a>, 
                                    <a href='https://huggingface.co/SG161222/Realistic_Vision_V3.0_VAE' target='_blank'>SG161222/Realistic_Vision_V3.0_VAE</a>, 
                                    <a href='https://huggingface.co/digiplay/AbsoluteReality_v1.8.1' target='_blank'>digiplay/AbsoluteReality_v1.8.1</a>, 
                                    <a href='https://huggingface.co/runwayml/stable-diffusion-v1-5' target='_blank'>runwayml/stable-diffusion-v1-5</a>, 
                                    <a href='https://huggingface.co/nitrosocke/Ghibli-Diffusion' target='_blank'>nitrosocke/Ghibli-Diffusion</a></br>
                                    """
                                    #                                 <a href='https://huggingface.co/ckpt/anything-v4.5-vae-swapped' target='_blank'>ckpt/anything-v4.5-vae-swapped</a>,
                                )
                            with gr.Box():
                                gr.HTML(
                                    """
                                    <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                    <div style='text-align: justified'>
                                    <b>Usage :</b></br>
                                    - (optional) Modify the settings to use another model, modify the number of frames to generate or change dimensions of the outputs</br>
                                    - Fill the <b>prompt</b> with what you want to see in your output video</br>
                                    - Fill the <b>negative prompt</b> with what you DO NOT want to see in your output video</br>
                                    - Click the <b>Generate</b> button</br>
                                    - After generation, generated video is displayed in the <b>Generated video</b> field.
                                    </br>
                                    <b>Models :</b></br>
                                    - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or  <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors models in the directory /models/Stable Diffusion. Restart to see them in the models list.
                                    </div>
                                    """
                                )
                        with gr.Accordion("Settings", open=True):
                            with gr.Row():
                                with gr.Column():
                                    model_animatediff_lcm = gr.Dropdown(choices=model_list_animatediff_lcm,
                                                                        value=model_list_animatediff_lcm[0],
                                                                        label="Model",
                                                                        info="Choose model to use for inference")
                                with gr.Column():
                                    model_adapters_animatediff_lcm = gr.Dropdown(
                                        choices=list(model_list_adapters_animatediff_lcm.keys()),
                                        value=list(model_list_adapters_animatediff_lcm.keys())[0],
                                        label="Adapter",
                                        info="Choose adapter model to use for inference")
                                with gr.Column():
                                    num_inference_step_animatediff_lcm = gr.Slider(1, webui_global_steps_max, step=1,
                                                                                   value=4,
                                                                                   label="Steps",
                                                                                   info="steps")
                                with gr.Column():
                                    sampler_animatediff_lcm = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()),
                                                                          value="LCM", label="sampler",
                                                                          info="sampler",
                                                                          interactive=True)
                            with gr.Row():
                                with gr.Column():
                                    adapter_animatediff_lcm = gr.Dropdown(choices=adapter_list_animatediff_lcm,
                                                                          value=adapter_list_animatediff_lcm[0],
                                                                          label="adapter",
                                                                          info="Choose adapter to use for inference")
                            with gr.Row():
                                with gr.Column():
                                    lora_animatediff_lcm = gr.Dropdown(choices=lora_list_animatediff_lcm,
                                                                       value=lora_list_animatediff_lcm[0],
                                                                       label="Lora",
                                                                       info="Choose Lora to use for inference")
                            with gr.Row():
                                with gr.Column():
                                    num_inference_step_animatediff_lcm = gr.Slider(1, 100, step=1, value=4,
                                                                                   label="Steps",
                                                                                   info="Number of iterations per video. Results and speed depends of sampler")
                                with gr.Column():
                                    sampler_animatediff_lcm = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()),
                                                                          value="LCM", label="Sampler",
                                                                          info="Sampler to use for inference",
                                                                          interactive=False)
                            with gr.Row():
                                with gr.Column():
                                    guidance_scale_animatediff_lcm = gr.Slider(0.1, 20.0, step=0.1, value=2.0,
                                                                               label="CFG scale",
                                                                               info="Low values : more creativity. High values : more fidelity to the prompts")
                                with gr.Column():
                                    seed_animatediff_lcm = gr.Slider(0, 10000000000, step=1, value=0,
                                                                     label="Seed(0 for random)",
                                                                     info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                                with gr.Column():
                                    num_frames_animatediff_lcm = gr.Slider(1, 1200, step=1, value=16,
                                                                           label="Video Length (frames)",
                                                                           info="Number of frames in the output video (@8fps)")
                                with gr.Column():
                                    num_fps_animatediff_lcm = gr.Slider(1, 120, step=1, value=8,
                                                                        label="fps",
                                                                        info="fps")
                            with gr.Row():
                                with gr.Column():
                                    width_animatediff_lcm = gr.Slider(128, 1280, step=64, value=512,
                                                                      label="Video Width", info="Width of outputs")
                                with gr.Column():
                                    height_animatediff_lcm = gr.Slider(128, 1280, step=64, value=512,
                                                                       label="Video Height",
                                                                       info="Height of outputs")
                                with gr.Column():
                                    num_videos_per_prompt_animatediff_lcm = gr.Slider(1, 4, step=1, value=1,
                                                                                      label="Batch size",
                                                                                      info="Number of videos to generate in a single run",
                                                                                      interactive=False)
                            with gr.Row():
                                with gr.Column():
                                    num_prompt_animatediff_lcm = gr.Slider(1, 32, step=1, value=1,
                                                                           label="Batch count",
                                                                           info="Number of batch to run successively")
                                with gr.Column():
                                    use_gfpgan_animatediff_lcm = gr.Checkbox(value=True,
                                                                             label="Use GFPGAN to restore faces",
                                                                             info="Use GFPGAN to enhance faces in the outputs",
                                                                             visible=True)
                                with gr.Column():
                                    tkme_animatediff_lcm = gr.Slider(0.0, 1.0, step=0.01, value=0,
                                                                     label="Token Merging ratio",
                                                                     info="0=slow,best quality, 1=fast,worst quality",
                                                                     visible=True)
                            with gr.Row():
                                with gr.Column():
                                    save_ini_btn_animatediff_lcm = gr.Button("Save custom defaults settings 💾")
                                with gr.Column():
                                    module_name_animatediff_lcm = gr.Textbox(value="animatediff_lcm", visible=False,
                                                                             interactive=False)
                                    del_ini_btn_animatediff_lcm = gr.Button("Delete custom defaults settings 🗑️",
                                                                            interactive=True if test_cfg_exist(
                                                                                module_name_animatediff_lcm.value) else False)
                                    save_ini_btn_animatediff_lcm.click(
                                        fn=write_ini,
                                        inputs=[
                                            module_name_animatediff_lcm,
                                            model_animatediff_lcm,
                                            adapter_animatediff_lcm,
                                            lora_animatediff_lcm,
                                            num_inference_step_animatediff_lcm,
                                            sampler_animatediff_lcm,
                                            guidance_scale_animatediff_lcm,
                                            seed_animatediff_lcm,
                                            num_frames_animatediff_lcm,
                                            width_animatediff_lcm,
                                            height_animatediff_lcm,
                                            num_videos_per_prompt_animatediff_lcm,
                                            num_prompt_animatediff_lcm,
                                            use_gfpgan_animatediff_lcm,
                                            tkme_animatediff_lcm,
                                        ]
                                    )
                                    save_ini_btn_animatediff_lcm.click(fn=lambda: gr.Info('Settings saved'))
                                    save_ini_btn_animatediff_lcm.click(
                                        fn=lambda: del_ini_btn_animatediff_lcm.update(interactive=True),
                                        outputs=del_ini_btn_animatediff_lcm)
                                    del_ini_btn_animatediff_lcm.click(
                                        fn=lambda: del_ini(module_name_animatediff_lcm.value))
                                    del_ini_btn_animatediff_lcm.click(fn=lambda: gr.Info('Settings deleted'))
                                    del_ini_btn_animatediff_lcm.click(
                                        fn=lambda: del_ini_btn_animatediff_lcm.update(interactive=False),
                                        outputs=del_ini_btn_animatediff_lcm)
                            if test_cfg_exist(module_name_animatediff_lcm.value):
                                readcfg_animatediff_lcm = read_ini_animatediff_lcm(
                                    module_name_animatediff_lcm.value)
                                model_animatediff_lcm.value = readcfg_animatediff_lcm[0]
                                adapter_animatediff_lcm.value = readcfg_animatediff_lcm[1]
                                lora_animatediff_lcm.value = readcfg_animatediff_lcm[2]
                                num_inference_step_animatediff_lcm.value = readcfg_animatediff_lcm[3]
                                sampler_animatediff_lcm.value = readcfg_animatediff_lcm[4]
                                guidance_scale_animatediff_lcm.value = readcfg_animatediff_lcm[5]
                                seed_animatediff_lcm.value = readcfg_animatediff_lcm[6]
                                num_frames_animatediff_lcm.value = readcfg_animatediff_lcm[7]
                                width_animatediff_lcm.value = readcfg_animatediff_lcm[8]
                                height_animatediff_lcm.value = readcfg_animatediff_lcm[9]
                                num_videos_per_prompt_animatediff_lcm.value = readcfg_animatediff_lcm[10]
                                num_prompt_animatediff_lcm.value = readcfg_animatediff_lcm[11]
                                use_gfpgan_animatediff_lcm.value = readcfg_animatediff_lcm[12]
                                tkme_animatediff_lcm.value = readcfg_animatediff_lcm[13]
                        with gr.Row():
                            with gr.Column(scale=2):
                                with gr.Row():
                                    with gr.Column():
                                        prompt_animatediff_lcm = gr.Textbox(lines=5, max_lines=5, label="Prompt",
                                                                            info="Describe what you want in your video",
                                                                            placeholder="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution")
                                with gr.Row():
                                    with gr.Column():
                                        negative_prompt_animatediff_lcm = gr.Textbox(lines=5, max_lines=5,
                                                                                     label="Negative Prompt",
                                                                                     info="Describe what you DO NOT want in your video",
                                                                                     placeholder="bad quality, worst quality, low resolution")
                                with gr.Row():
                                    with gr.Column():
                                        output_type_animatediff_lcm = gr.Radio(choices=["mp4", "gif"], value="mp4",
                                                                               label="output type",
                                                                               info="output type")
                                    with gr.Column():
                                        gr.Number(visible=False)
                            model_animatediff_lcm.change(
                                fn=change_model_type_animatediff_lcm,
                                inputs=[model_animatediff_lcm],
                                outputs=[
                                    sampler_animatediff_lcm,
                                    width_animatediff_lcm,
                                    height_animatediff_lcm,
                                    num_inference_step_animatediff_lcm,
                                    guidance_scale_animatediff_lcm,
                                    negative_prompt_animatediff_lcm,
                                ]
                            )
                            with gr.Column(scale=1):
                                out_animatediff_lcm = gr.Video(label="Generated video", height=400, visible=True,
                                                               interactive=False)
                                gif_out_animatediff_lcm = gr.Gallery(
                                    label="Generated gif",
                                    show_label=True,
                                    elem_id="gallery",
                                    columns=3,
                                    height=400,
                                    visible=False
                                )
                        with gr.Row():
                            with gr.Column():
                                btn_animatediff_lcm = gr.Button("Generate 🚀", variant="primary", visible=True)
                                btn_animatediff_lcm_gif = gr.Button("Generate 🚀", variant="primary", visible=False)
                            with gr.Column():
                                btn_animatediff_lcm_cancel = gr.Button("Cancel 🛑", variant="stop")
                                btn_animatediff_lcm_cancel.click(fn=initiate_stop_animatediff_lcm, inputs=None,
                                                                 outputs=None)
                            with gr.Column():
                                btn_animatediff_lcm_clear_input = gr.ClearButton(
                                    components=[prompt_animatediff_lcm, negative_prompt_animatediff_lcm],
                                    value="Clear inputs 🧹")
                            with gr.Column():
                                btn_animatediff_lcm_clear_output = gr.ClearButton(
                                    components=[out_animatediff_lcm, gif_out_animatediff_lcm], value="Clear outputs 🧹")
                                btn_animatediff_lcm.click(
                                    fn=video_animatediff_lcm,
                                    inputs=[
                                        model_animatediff_lcm,
                                        model_adapters_animatediff_lcm,
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
                                    ],
                                    outputs=out_animatediff_lcm,
                                    show_progress="full",
                                )
                                btn_animatediff_lcm_gif.click(
                                    fn=video_animatediff_lcm,
                                    inputs=[
                                        model_animatediff_lcm,
                                        model_adapters_animatediff_lcm,
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
                                    ],
                                    outputs=gif_out_animatediff_lcm,
                                    show_progress="full",
                                )
                                output_type_animatediff_lcm.change(
                                    fn=change_output_type_animatediff_lcm,
                                    inputs=[
                                        output_type_animatediff_lcm,
                                    ],
                                    outputs=[
                                        out_animatediff_lcm,
                                        gif_out_animatediff_lcm,
                                        btn_animatediff_lcm,
                                        btn_animatediff_lcm_gif,
                                    ]
                                )

                    if ram_size() >= 16:
                        titletab_tab_animatediff_lightning = "Animate Lightning 📼"
                    else:
                        titletab_tab_animatediff_lightning = "Animate Lightning ⛔"
                    with gr.TabItem(titletab_tab_animatediff_lightning, id=143) as tab_animatediff_lightning:
                        with gr.Accordion("About", open=False):
                            with gr.Box():
                                gr.HTML(
                                    """
                                    <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                    <b>Module : </b>Animate Lightning</br>
                                    <b>Function : </b>Generate video from a prompt and a negative prompt using <a href='https://hf-mirror.com/ByteDance/AnimateDiff-Lightning/' target='_blank'>Animate Lightning</a> with <a href='https://stability.ai/stablediffusion' target='_blank'>Stable Diffusion</a> Models</br>
                                    <b>Input(s) : </b>Prompt, negative prompt</br>
                                    <b>Output(s) : </b>Video</br>
                                    <b>HF model page : </b>
                                    <a href='https://huggingface.co/emilianJR/epiCRealism' target='_blank'>emilianJR/epiCRealism</a>, 
                                    <a href='https://huggingface.co/SG161222/Realistic_Vision_V3.0_VAE' target='_blank'>SG161222/Realistic_Vision_V3.0_VAE</a>, 
                                    <a href='https://huggingface.co/nitrosocke/Ghibli-Diffusion' target='_blank'>nitrosocke/Ghibli-Diffusion</a></br>
                                    """
                                )
                            with gr.Box():
                                gr.HTML(
                                    """
                                    <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                    <div style='text-align: justified'>
                                    <b>Usage :</b></br>
                                    - (optional) Modify the settings to use another model, modify the number of frames to generate or change dimensions of the outputs</br>
                                    - Fill the <b>prompt</b> with what you want to see in your output video</br>
                                    - Fill the <b>negative prompt</b> with what you DO NOT want to see in your output video</br>
                                    - Click the <b>Generate</b> button</br>
                                    - After generation, generated video is displayed in the <b>Generated video</b> field.
                                    </br>
                                    <b>Models :</b></br>
                                    - You could place <a href='https://huggingface.co/' target='_blank'>huggingface.co</a> or  <a href='https://www.civitai.com/' target='_blank'>civitai.com</a> Stable diffusion based safetensors models in the directory /models/Stable Diffusion. Restart to see them in the models list.
                                    </div>
                                    """
                                )
                        with gr.Accordion("Settings", open=True):
                            with gr.Row():
                                with gr.Column():
                                    model_animatediff_lightning = gr.Dropdown(choices=model_list_animatediff_lightning,
                                                                              value=model_list_animatediff_lightning[0],
                                                                              label="Model",
                                                                              info="Choose model to use for inference")
                            with gr.Row():
                                with gr.Column():
                                    adapter_animatediff_lightning = gr.Dropdown(
                                        choices=adapter_list_animatediff_lightning,
                                        value=adapter_list_animatediff_lightning[0],
                                        label="adapter",
                                        info="Choose adapter to use for inference")
                            # with gr.Row():
                            #     with gr.Column():
                            #         lora_animatediff_lightning = gr.Dropdown(choices=lora_list_animatediff_lightning,
                            #                                                  value=lora_list_animatediff_lightning[0],
                            #                                                  label="Lora",
                            #                                                  info="Choose Lora to use for inference")
                            with gr.Row():
                                with gr.Column():
                                    num_inference_step_animatediff_lightning = gr.Slider(1, 100, step=1, value=4,
                                                                                         label="Steps",
                                                                                         info="Number of iterations per video. Results and speed depends of sampler")
                                with gr.Column():
                                    sampler_animatediff_lightning = gr.Dropdown(choices=list(SCHEDULER_MAPPING.keys()),
                                                                                value="Euler", label="Sampler",
                                                                                info="Sampler to use for inference",
                                                                                interactive=False)
                            with gr.Row():
                                with gr.Column():
                                    guidance_scale_animatediff_lightning = gr.Slider(0.1, 20.0, step=0.1, value=2.0,
                                                                                     label="CFG scale",
                                                                                     info="Low values : more creativity. High values : more fidelity to the prompts")
                                with gr.Column():
                                    seed_animatediff_lightning = gr.Slider(0, 10000000000, step=1, value=0,
                                                                           label="Seed(0 for random)",
                                                                           info="Seed to use for generation. Depending on scheduler, may permit reproducibility")
                                with gr.Column():
                                    num_frames_animatediff_lightning = gr.Slider(1, 1200, step=1, value=16,
                                                                                 label="Video Length (frames)",
                                                                                 info="Number of frames in the output video (@8fps)")
                            with gr.Row():
                                with gr.Column():
                                    width_animatediff_lightning = gr.Slider(128, 1280, step=64, value=512,
                                                                            label="Video Width",
                                                                            info="Width of outputs")
                                with gr.Column():
                                    height_animatediff_lightning = gr.Slider(128, 1280, step=64, value=512,
                                                                             label="Video Height",
                                                                             info="Height of outputs")
                                with gr.Column():
                                    num_videos_per_prompt_animatediff_lightning = gr.Slider(1, 4, step=1, value=1,
                                                                                            label="Batch size",
                                                                                            info="Number of videos to generate in a single run",
                                                                                            interactive=False)
                            with gr.Row():
                                with gr.Column():
                                    num_prompt_animatediff_lightning = gr.Slider(1, 32, step=1, value=1,
                                                                                 label="Batch count",
                                                                                 info="Number of batch to run successively")
                                with gr.Column():
                                    use_gfpgan_animatediff_lightning = gr.Checkbox(value=True,
                                                                                   label="Use GFPGAN to restore faces",
                                                                                   info="Use GFPGAN to enhance faces in the outputs",
                                                                                   visible=True)
                                with gr.Column():
                                    tkme_animatediff_lightning = gr.Slider(0.0, 1.0, step=0.01, value=0,
                                                                           label="Token Merging ratio",
                                                                           info="0=slow,best quality, 1=fast,worst quality",
                                                                           visible=True)
                            with gr.Row():
                                with gr.Column():
                                    save_ini_btn_animatediff_lightning = gr.Button("Save custom defaults settings 💾")
                                with gr.Column():
                                    module_name_animatediff_lightning = gr.Textbox(value="animatediff_lightning",
                                                                                   visible=False,
                                                                                   interactive=False)
                                    del_ini_btn_animatediff_lightning = gr.Button("Delete custom defaults settings 🗑️",
                                                                                  interactive=True if test_cfg_exist(
                                                                                      module_name_animatediff_lightning.value) else False)
                                    save_ini_btn_animatediff_lightning.click(
                                        fn=write_ini,
                                        inputs=[
                                            module_name_animatediff_lightning,
                                            model_animatediff_lightning,
                                            adapter_animatediff_lightning,
                                            # lora_animatediff_lightning,
                                            num_inference_step_animatediff_lightning,
                                            sampler_animatediff_lightning,
                                            guidance_scale_animatediff_lightning,
                                            seed_animatediff_lightning,
                                            num_frames_animatediff_lightning,
                                            width_animatediff_lightning,
                                            height_animatediff_lightning,
                                            num_videos_per_prompt_animatediff_lightning,
                                            num_prompt_animatediff_lightning,
                                            use_gfpgan_animatediff_lightning,
                                            tkme_animatediff_lightning,
                                        ]
                                    )
                                    save_ini_btn_animatediff_lightning.click(fn=lambda: gr.Info('Settings saved'))
                                    save_ini_btn_animatediff_lightning.click(
                                        fn=lambda: del_ini_btn_animatediff_lightning.update(interactive=True),
                                        outputs=del_ini_btn_animatediff_lightning)
                                    del_ini_btn_animatediff_lightning.click(
                                        fn=lambda: del_ini(module_name_animatediff_lightning.value))
                                    del_ini_btn_animatediff_lightning.click(fn=lambda: gr.Info('Settings deleted'))
                                    del_ini_btn_animatediff_lightning.click(
                                        fn=lambda: del_ini_btn_animatediff_lightning.update(interactive=False),
                                        outputs=del_ini_btn_animatediff_lightning)
                            if test_cfg_exist(module_name_animatediff_lightning.value):
                                readcfg_animatediff_lightning = read_ini_animatediff_lightning(
                                    module_name_animatediff_lightning.value)
                                model_animatediff_lightning.value = readcfg_animatediff_lightning[0]
                                adapter_animatediff_lightning.value = readcfg_animatediff_lightning[1]
                                # lora_animatediff_lightning.value = readcfg_animatediff_lightning[2]
                                num_inference_step_animatediff_lightning.value = readcfg_animatediff_lightning[2]
                                sampler_animatediff_lightning.value = readcfg_animatediff_lightning[3]
                                guidance_scale_animatediff_lightning.value = readcfg_animatediff_lightning[4]
                                seed_animatediff_lightning.value = readcfg_animatediff_lightning[5]
                                num_frames_animatediff_lightning.value = readcfg_animatediff_lightning[6]
                                width_animatediff_lightning.value = readcfg_animatediff_lightning[7]
                                height_animatediff_lightning.value = readcfg_animatediff_lightning[8]
                                num_videos_per_prompt_animatediff_lightning.value = readcfg_animatediff_lightning[9]
                                num_prompt_animatediff_lightning.value = readcfg_animatediff_lightning[10]
                                use_gfpgan_animatediff_lightning.value = readcfg_animatediff_lightning[11]
                                tkme_animatediff_lightning.value = readcfg_animatediff_lightning[12]
                        with gr.Row():
                            with gr.Column(scale=2):
                                with gr.Row():
                                    with gr.Column():
                                        prompt_animatediff_lightning = gr.Textbox(lines=5, max_lines=5, label="Prompt",
                                                                                  info="Describe what you want in your video",
                                                                                  placeholder="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution")
                                with gr.Row():
                                    with gr.Column():
                                        negative_prompt_animatediff_lightning = gr.Textbox(lines=5, max_lines=5,
                                                                                           label="Negative Prompt",
                                                                                           info="Describe what you DO NOT want in your video",
                                                                                           placeholder="bad quality, worst quality, low resolution")
                            model_animatediff_lightning.change(
                                fn=change_model_type_animatediff_lightning,
                                inputs=[model_animatediff_lightning],
                                outputs=[
                                    sampler_animatediff_lightning,
                                    width_animatediff_lightning,
                                    height_animatediff_lightning,
                                    num_inference_step_animatediff_lightning,
                                    guidance_scale_animatediff_lightning,
                                    negative_prompt_animatediff_lightning,
                                ]
                            )
                            with gr.Column(scale=1):
                                out_animatediff_lightning = gr.Video(label="Generated video", height=400,
                                                                     interactive=False)
                        with gr.Row():
                            btn_animatediff_lightning = gr.Button("Generate 🚀", variant="primary")
                            btn_animatediff_lightning_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_animatediff_lightning_cancel.click(fn=initiate_stop_animatediff_lightning, inputs=None,
                                                                   outputs=None)
                            btn_animatediff_lightning_clear_input = gr.ClearButton(
                                components=[prompt_animatediff_lightning, negative_prompt_animatediff_lightning],
                                value="Clear inputs 🧹")
                            btn_animatediff_lightning_clear_output = gr.ClearButton(
                                components=[out_animatediff_lightning],
                                value="Clear outputs 🧹")
                            btn_animatediff_lightning.click(
                                fn=video_animatediff_lightning,
                                inputs=[
                                    model_animatediff_lightning,
                                    adapter_animatediff_lightning,
                                    # lora_animatediff_lightning,
                                    num_inference_step_animatediff_lightning,
                                    sampler_animatediff_lightning,
                                    guidance_scale_animatediff_lightning,
                                    seed_animatediff_lightning,
                                    num_frames_animatediff_lightning,
                                    height_animatediff_lightning,
                                    width_animatediff_lightning,
                                    num_videos_per_prompt_animatediff_lightning,
                                    num_prompt_animatediff_lightning,
                                    prompt_animatediff_lightning,
                                    negative_prompt_animatediff_lightning,
                                    nsfw_filter,
                                    use_gfpgan_animatediff_lightning,
                                    tkme_animatediff_lightning,
                                ],
                                outputs=out_animatediff_lightning,
                                show_progress="full",
                            )

            ip_advanced.change(lambda: None, queue=False, show_progress=False, _js=down_js)

            # current_tab = gr.State(value='uov')
            current_tab = gr.Textbox(value='uov', visible=False)
            # default_image = gr.State(value=None)

            # lambda_img = lambda x: x['image'] if isinstance(x, dict) else x
            # uov_input_image.upload(lambda_img, inputs=uov_input_image, outputs=default_image, queue=False)
            # inpaint_input_image.upload(lambda_img, inputs=inpaint_input_image, outputs=default_image, queue=False)

            with gr.Row(elem_classes='advanced_check_row'):
                audio_factory_checkbox = gr.Checkbox(label='Audio-Factory', value=False, container=True,
                                                     info="| text-to-music | audio-to-music | text-to-speech |",
                                                     elem_classes='min_check')
            with gr.Row(visible=False) as audio_input_panel:
                with gr.Tabs():
                    if ram_size() >= 16:
                        titletab_chattts_mel = "ChatTTS 🎶"
                    else:
                        titletab_chattts_mel = "ChatTTS ⛔"
                    with gr.TabItem(titletab_chattts_mel, id=132) as tab_chattts_mel:
                        from resources.chatTTS.webui.wording import get
                        import resources.chatTTS.webui.batch_option
                        import resources.chatTTS.webui.text_options
                        import resources.chatTTS.webui.seed_option
                        import resources.chatTTS.webui.aduio_option
                        import resources.chatTTS.webui.enhance_option
                        import resources.chatTTS.webui.output_option
                        import resources.chatTTS.webui.config_option

                        with gr.Accordion("About", open=False):
                            with gr.Box():
                                gr.HTML(
                                    """
                                    <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                    <b>Module : </b>ChatTTS</br>
                                    <b>Function : </b>ChatTTS is a text-to-speech model designed specifically for dialogue scenarios such as LLM assistant. <a href='https://github.com/2noise/ChatTTS/tree/main/ChatTTS' target='_blank'>ChatTTS</a></br>
                                    <b>Input(s) : </b>Input prompt, Input audio</br>
                                    <b>Output(s) : </b>Generated audio</br>
                                    <b>HF model page : </b>
                                    <a href='https://huggingface.co/2Noise/ChatTTS' target='_blank'>2Noise/ChatTTS</a></br>
                                    """
                                )
                            with gr.Box():
                                gr.HTML(
                                    """
                                    <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                    <div style='text-align: justified'>
                                    <b>Usage :</b></br>
                                    - Select an audio source type (file or micro recording)</br>
                                    - Select an audio source by choosing a file or recording something</br>
                                    - Fill the <b>prompt</b> by describing the audio you want to generate from the text</br>
                                    - (optional) Modify the settings to change audio duration or inferences parameters</br>
                                    - Click the <b>Generate<b> button</br>
                                    - After generation, generated audio is available to listen in the <b>Generated audio<b> field.
                                    </div>
                                    """
                                )
                        with gr.Accordion("Settings", open=False):
                            with gr.Row():
                                with gr.Column():
                                    with gr.Row():
                                        gr.Markdown(get('TextOptionsTitle'))
                                    resources.chatTTS.webui.text_options.render()
                                    with gr.Row():
                                        gr.Markdown(get('SeedOptionsTitle'))
                                    resources.chatTTS.webui.seed_option.render()
                                    with gr.Row():
                                        gr.Markdown(get('AudioOptionsTitle'))
                                    resources.chatTTS.webui.aduio_option.render()
                                    with gr.Row():
                                        gr.Markdown(get('AudioEnhancementTitle'))
                                    resources.chatTTS.webui.enhance_option.render()
                                    with gr.Row():
                                        gr.Markdown(get('configmanager'))
                                    resources.chatTTS.webui.config_option.render()
                        with gr.Row():
                            resources.chatTTS.webui.batch_option.render()
                        with gr.Row():
                            resources.chatTTS.webui.output_option.render()

                        resources.chatTTS.webui.batch_option.listen()
                        resources.chatTTS.webui.text_options.listen()
                        resources.chatTTS.webui.seed_option.listen()
                        resources.chatTTS.webui.aduio_option.listen()
                        resources.chatTTS.webui.enhance_option.listen()
                        resources.chatTTS.webui.output_option.listen()

                    if ram_size() >= 16:
                        titletab_musicgen_mel = "MusicGen Melody 🎶"
                    else:
                        titletab_musicgen_mel = "MusicGen Melody ⛔"
                    with gr.TabItem(titletab_musicgen_mel, id=32) as tab_musicgen_mel:
                        with gr.Accordion("About", open=False):
                            with gr.Box():
                                gr.HTML(
                                    """
                                    <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                    <b>Module : </b>MusicGen Melody</br>
                                    <b>Function : </b>Generate music from a prompt with guidance from an input audio, using <a href='https://github.com/facebookresearch/audiocraft' target='_blank'>MusicGen</a></br>
                                    <b>Input(s) : </b>Input prompt, Input audio</br>
                                    <b>Output(s) : </b>Generated music</br>
                                    <b>HF model page : </b>
                                    <a href='https://huggingface.co/facebook/musicgen-melody' target='_blank'>facebook/musicgen-melody</a></br>
                                    """
                                )
                            with gr.Box():
                                gr.HTML(
                                    """
                                    <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                    <div style='text-align: justified'>
                                    <b>Usage :</b></br>
                                    - Select an audio source type (file or micro recording)</br>
                                    - Select an audio source by choosing a file or recording something</br>
                                    - Fill the <b>prompt</b> by describing the music you want to generate from the audio source</br>
                                    - (optional) Modify the settings to change audio duration or inferences parameters</br>
                                    - Click the <b>Generate<b> button</br>
                                    - After generation, generated music is available to listen in the <b>Generated music<b> field.
                                    </div>
                                    """
                                )
                        with gr.Accordion("Settings", open=False):
                            with gr.Row():
                                with gr.Column():
                                    model_musicgen_mel = gr.Dropdown(choices=modellist_musicgen_mel,
                                                                     value=modellist_musicgen_mel[0], label="Model",
                                                                     info="Choose model to use for inference")
                                with gr.Column():
                                    duration_musicgen_mel = gr.Slider(1, 160, step=1, value=5,
                                                                      label="Audio length (sec)")
                                with gr.Column():
                                    cfg_coef_musicgen_mel = gr.Slider(0.1, 20.0, step=0.1, value=3.0,
                                                                      label="CFG scale",
                                                                      info="Low values : more creativity. High values : more fidelity to the prompts")
                                with gr.Column():
                                    num_batch_musicgen_mel = gr.Slider(1, 32, step=1, value=1, label="Batch count",
                                                                       info="Number of batch to run successively")
                            with gr.Row():
                                with gr.Column():
                                    use_sampling_musicgen_mel = gr.Checkbox(value=True, label="Use sampling")
                                with gr.Column():
                                    temperature_musicgen_mel = gr.Slider(0.0, 10.0, step=0.1, value=1.0,
                                                                         label="temperature")
                                with gr.Column():
                                    top_k_musicgen_mel = gr.Slider(0, 500, step=1, value=250, label="top_k")
                                with gr.Column():
                                    top_p_musicgen_mel = gr.Slider(0.0, 500.0, step=1.0, value=0.0, label="top_p")
                            with gr.Row():
                                with gr.Column():
                                    save_ini_btn_musicgen_mel = gr.Button("Save custom defaults settings 💾")
                                with gr.Column():
                                    module_name_musicgen_mel = gr.Textbox(value="musicgen_mel", visible=False,
                                                                          interactive=False)
                                    del_ini_btn_musicgen_mel = gr.Button("Delete custom defaults settings 🗑️",
                                                                         interactive=True if test_cfg_exist(
                                                                             module_name_musicgen_mel.value) else False)
                                    save_ini_btn_musicgen_mel.click(
                                        fn=write_ini,
                                        inputs=[
                                            module_name_musicgen_mel,
                                            model_musicgen_mel,
                                            duration_musicgen_mel,
                                            cfg_coef_musicgen_mel,
                                            num_batch_musicgen_mel,
                                            use_sampling_musicgen_mel,
                                            temperature_musicgen_mel,
                                            top_k_musicgen_mel,
                                            top_p_musicgen_mel,
                                        ]
                                    )
                                    save_ini_btn_musicgen_mel.click(fn=lambda: gr.Info('Settings saved'))
                                    save_ini_btn_musicgen_mel.click(
                                        fn=lambda: del_ini_btn_musicgen_mel.update(interactive=True),
                                        outputs=del_ini_btn_musicgen_mel)
                                    del_ini_btn_musicgen_mel.click(
                                        fn=lambda: del_ini(module_name_musicgen_mel.value))
                                    del_ini_btn_musicgen_mel.click(fn=lambda: gr.Info('Settings deleted'))
                                    del_ini_btn_musicgen_mel.click(
                                        fn=lambda: del_ini_btn_musicgen_mel.update(interactive=False),
                                        outputs=del_ini_btn_musicgen_mel)
                            if test_cfg_exist(module_name_musicgen_mel.value):
                                readcfg_musicgen_mel = read_ini_musicgen_mel(module_name_musicgen_mel.value)
                                model_musicgen_mel.value = readcfg_musicgen_mel[0]
                                duration_musicgen_mel.value = readcfg_musicgen_mel[1]
                                cfg_coef_musicgen_mel.value = readcfg_musicgen_mel[2]
                                num_batch_musicgen_mel.value = readcfg_musicgen_mel[3]
                                use_sampling_musicgen_mel.value = readcfg_musicgen_mel[4]
                                temperature_musicgen_mel.value = readcfg_musicgen_mel[5]
                                top_k_musicgen_mel.value = readcfg_musicgen_mel[6]
                                top_p_musicgen_mel.value = readcfg_musicgen_mel[7]
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    source_type_musicgen_mel = gr.Radio(choices=["audio", "micro"], value="audio",
                                                                        label="Source audio type",
                                                                        info="Choose source audio type")
                        with gr.Row(equal_height=True):
                            with gr.Column():
                                source_audio_musicgen_mel = gr.Audio(label="Source audio", source="upload",
                                                                     type="filepath")
                                source_type_musicgen_mel.change(fn=change_source_type_musicgen_mel,
                                                                inputs=source_type_musicgen_mel,
                                                                outputs=source_audio_musicgen_mel)
                            with gr.Column():
                                prompt_musicgen_mel = gr.Textbox(label="Describe your music", lines=8, max_lines=8,
                                                                 placeholder="90s rock song with loud guitars and heavy drums")
                            with gr.Column():
                                out_musicgen_mel = gr.Audio(label="Generated music", type="filepath",
                                                            show_download_button=True, interactive=False)
                        with gr.Row():
                            btn_musicgen_mel = gr.Button("Generate 🚀", variant="primary")
                            btn_musicgen_mel_cancel = gr.Button("Cancel 🛑", variant="stop")
                            btn_musicgen_mel_cancel.click(fn=initiate_stop_musicgen_mel, inputs=None, outputs=None)
                            btn_musicgen_mel_clear_input = gr.ClearButton(
                                components=[prompt_musicgen_mel, source_audio_musicgen_mel], value="Clear inputs 🧹")
                            btn_musicgen_mel_clear_output = gr.ClearButton(components=out_musicgen_mel,
                                                                           value="Clear outputs 🧹")
                            btn_musicgen_mel.click(
                                fn=music_musicgen_mel,
                                inputs=[
                                    prompt_musicgen_mel,
                                    model_musicgen_mel,
                                    duration_musicgen_mel,
                                    num_batch_musicgen_mel,
                                    temperature_musicgen_mel,
                                    top_k_musicgen_mel,
                                    top_p_musicgen_mel,
                                    use_sampling_musicgen_mel,
                                    cfg_coef_musicgen_mel,
                                    source_audio_musicgen_mel,
                                    source_type_musicgen_mel,
                                ],
                                outputs=out_musicgen_mel,
                                show_progress="full",
                            )

                    with gr.TabItem("Bark 🗣️", id=36) as tab_bark:
                        with gr.Accordion("About", open=False):
                            with gr.Box():
                                gr.HTML(
                                    """
                                    <h1 style='text-align: left'; text-decoration: underline;>Informations</h1>
                                    <b>Module : </b>Bark</br>
                                    <b>Function : </b>Generate high quality text-to-speech in several languages with <a href='https://github.com/suno-ai/bark' target='_blank'>Bark</a></br>
                                    <b>Input(s) : </b>Prompt</br>
                                    <b>Output(s) : </b>Generated speech</br>
                                    <b>HF model page : </b>
                                    <a href='https://huggingface.co/suno/bark' target='_blank'>suno/bark</a> ,
                                    <a href='https://huggingface.co/suno/bark-small' target='_blank'>suno/bark-small</a></br>               
                                    """
                                )
                            with gr.Box():
                                gr.HTML(
                                    """
                                    <h1 style='text-align: left'; text-decoration: underline;>Help</h1>
                                    <div style='text-align: justified'>
                                    <b>Usage :</b></br>
                                    - Fill the <b>prompt</b> with the text you want to hear</br>                                
                                    - (optional) Modify the settings to select a model and a voice</br>                                
                                    - Click the <b>Generate</b> button</br>
                                    - After generation, generated audio is available to listen in the <b>Generated speech</b> field.</br>
                                    <b>Tips : </b>You can add modifications to the generated voices, by adding the following in your prompts : 
                                    [laughter]</br>
                                    [laughs]</br>
                                    [sighs]</br>
                                    [music]</br>
                                    [gasps]</br>
                                    [clears throat]</br>
                                    — or ... for hesitations</br>
                                    ♪ for song lyrics</br>
                                    CAPITALIZATION for emphasis of a word</br>
                                    [MAN] and [WOMAN] to bias Bark toward male and female speakers, respectively</br>
                                    </div>
                                    """
                                )
                        with gr.Accordion("Settings", open=False):
                            with gr.Row():
                                with gr.Column():
                                    model_bark = gr.Dropdown(choices=model_list_bark, value=model_list_bark[0],
                                                             label="Model",
                                                             info="Choose model to use for inference")
                                with gr.Column():
                                    voice_preset_bark = gr.Dropdown(choices=list(voice_preset_list_bark.keys()),
                                                                    value=list(voice_preset_list_bark.keys())[2],
                                                                    label="Voice")
                            with gr.Row():
                                with gr.Column():
                                    save_ini_btn_bark = gr.Button("Save custom defaults settings 💾")
                                with gr.Column():
                                    module_name_bark = gr.Textbox(value="bark", visible=False, interactive=False)
                                    del_ini_btn_bark = gr.Button("Delete custom defaults settings 🗑️",
                                                                 interactive=True if test_cfg_exist(
                                                                     module_name_bark.value) else False)
                                    save_ini_btn_bark.click(
                                        fn=write_ini,
                                        inputs=[
                                            module_name_bark,
                                            model_bark,
                                            voice_preset_bark,
                                        ]
                                    )
                                    save_ini_btn_bark.click(fn=lambda: gr.Info('Settings saved'))
                                    save_ini_btn_bark.click(fn=lambda: del_ini_btn_bark.update(interactive=True),
                                                            outputs=del_ini_btn_bark)
                                    del_ini_btn_bark.click(fn=lambda: del_ini(module_name_bark.value))
                                    del_ini_btn_bark.click(fn=lambda: gr.Info('Settings deleted'))
                                    del_ini_btn_bark.click(fn=lambda: del_ini_btn_bark.update(interactive=False),
                                                           outputs=del_ini_btn_bark)
                            if test_cfg_exist(module_name_bark.value):
                                readcfg_bark = read_ini_bark(module_name_bark.value)
                                model_bark.value = readcfg_bark[0]
                                voice_preset_bark.value = readcfg_bark[1]
                        with gr.Row():
                            with gr.Column():
                                prompt_bark = gr.Textbox(label="Text to speech", lines=5, max_lines=10,
                                                         placeholder="Type or past here what you want to hear ...")
                            with gr.Column():
                                out_bark = gr.Audio(label="Generated speech", type="filepath",
                                                    show_download_button=True,
                                                    interactive=False)
                        with gr.Row():
                            with gr.Column():
                                btn_bark = gr.Button("Generate 🚀", variant="primary")
                            with gr.Column():
                                btn_bark_clear_input = gr.ClearButton(components=prompt_bark,
                                                                      value="Clear inputs 🧹")
                            with gr.Column():
                                btn_bark_clear_output = gr.ClearButton(components=out_bark, value="Clear outputs 🧹")
                            btn_bark.click(
                                fn=music_bark,
                                inputs=[
                                    prompt_bark,
                                    model_bark,
                                    voice_preset_bark,
                                ],
                                outputs=out_bark,
                                show_progress="full",
                            )


            def toggle_audio_file(choice):
                if not choice:
                    return gr.update(visible=True), gr.update(visible=False)
                else:
                    return gr.update(visible=False), gr.update(visible=True)


            def ref_video_fn(path_of_ref_video):
                if path_of_ref_video is not None:
                    return gr.update(value=True)
                else:
                    return gr.update(value=False)


            def update_video_factory(x):
                return gr.update(visible=x), gr.update(visible=x)


            def update_image_factory(x):
                return gr.update(visible=x)


            def update_text_factory(x):
                return gr.update(visible=x)


            def update_audio_factory(x):
                return gr.update(visible=x)


            image_factory_checkbox.change(update_image_factory, inputs=image_factory_checkbox,
                                          outputs=image_input_panel, queue=False, show_progress=False,
                                          _js=switch_js)

            video_factory_checkbox.change(update_video_factory, inputs=video_factory_checkbox,
                                          outputs=[video_input_panel, btn], queue=False, show_progress=False,
                                          _js=switch_js)

            text_factory_checkbox.change(update_text_factory, inputs=text_factory_checkbox,
                                         outputs=text_input_panel, queue=False, show_progress=False, _js=switch_js)

            audio_factory_checkbox.change(update_audio_factory, inputs=audio_factory_checkbox,
                                          outputs=audio_input_panel, queue=False, show_progress=False,
                                          _js=switch_js)


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


            uov_tab.select(lambda: 'uov', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            inpaint_tab.select(lambda: 'inpaint', outputs=current_tab, queue=False, _js=down_js,
                               show_progress=False)
            ip_tab.select(lambda: 'ip', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            desc_tab.select(lambda: 'desc', outputs=current_tab, queue=False, _js=down_js, show_progress=False)

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
                generate_button = gr.Button(label="Generate", value="Generate", elem_classes='type_row_half',
                                            elem_id='generate_button', visible=True)
                # load_parameter_button = gr.Button(label="Load Parameters", value="Load Parameters",
                #                                   elem_classes='type_row', elem_id='load_parameter_button',
                #                                   visible=False)
                skip_button = gr.Button(label="Skip", value="Skip", elem_classes='type_row_half', visible=False)
                stop_button = gr.Button(label="Stop", value="Stop", elem_classes='type_row_half',
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


                stop_button.click(stop_clicked, inputs=currentTask, outputs=currentTask, queue=True,
                                  every=1.0,
                                  show_progress=False, _js='cancelGenerateForever')
                skip_button.click(skip_clicked, inputs=currentTask, outputs=currentTask, queue=True,
                                  every=1.0,
                                  show_progress=False)

            with gr.Row(elem_classes='prompt_row'):
                prompt = gr.Textbox(label='Prompt', show_label=True, placeholder="Type prompt here.",
                                    container=True,
                                    autofocus=True, elem_classes='prompt_row', lines=5, max_lines=20,
                                    info='Describing objects that you DO want to see.',
                                    value=settings['prompt'])
            with gr.Row(elem_classes='n_negative_prompt_row'):
                negative_prompt = gr.Textbox(label='Negative Prompt', show_label=True,
                                             placeholder="Type negative prompt here.",
                                             info='Describing objects that you DO NOT want to see.', lines=5,
                                             max_lines=20)

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
                                                      show_edit_button=False)

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
                    load_last_prompt_button = gr.Button(label='Load Last Prompt', value='Load Last Prompt',
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
                        lambda: None, _js='()=>{refresh_style_localization();}')

                    gradio_receiver_style_selections.input(style_sorter.sort_styles,
                                                           inputs=style_selections,
                                                           outputs=style_selections,
                                                           queue=False,
                                                           show_progress=False).then(
                        lambda: None, _js='()=>{refresh_style_localization();}')


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
                            return gr.CheckboxGroup.update(show_label=False, container=False,
                                                           choices=legal_style_names,
                                                           value=default_legal_style_names,
                                                           label='Image Style')
                        elif choice == "ALL_Checked":
                            return gr.CheckboxGroup.update(show_label=False, container=False,
                                                           choices=legal_style_names,
                                                           value=legal_style_names,
                                                           label='Image Style')
                        elif choice == "ALL_UnChecked":
                            return gr.CheckboxGroup.update(show_label=False, container=False,
                                                           choices=legal_style_names,
                                                           value=[],
                                                           label='Image Style')
                        else:
                            special = _p(choice)
                            return gr.CheckboxGroup.update(show_label=False, container=False,
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
                        "ReMark": ""
                    }
                    if x is not None and "lora" in x.lower():
                        path = modules.config.paths_loras[0] + "\\"
                    else:
                        path = modules.config.paths_checkpoints[0] + "\\"
                    target_name = path + x + ".json"
                    if os.path.isfile(path=target_name):
                        with open(target_name, encoding='utf-8') as json_file:
                            try:
                                json_obj = json.load(json_file)
                                printF(name=MasterName.get_master_name(),
                                       info="[Parameters] json_obj = {}".format(json_obj)).printf()
                            except Exception as e:
                                printF(name=MasterName.get_master_name(),
                                       info="json -- get_target_info, e: {}".format(e)).printf()
                            finally:
                                json_file.close()
                                return json_obj
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
                    base_model = gr.Dropdown(label='Base Model (SDXL only)',
                                             choices=modules.config.sd_model_filenames,
                                             value=modules.config.default_base_model_name, show_label=True)

                    refiner_model = gr.Dropdown(label='Refiner (SDXL or SD 1.5)',
                                                choices=['None'] + modules.config.model_filenames,
                                                value=modules.config.default_refiner_model_name, show_label=True)
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


                def get_model_type_selector(x, y):
                    new_models_type_selectors_list = []
                    for msl in modules.config.model_types:
                        if x in msl:
                            new_models_type_selectors_list.append(msl)

                    new_model_type_filenames = ["None"]
                    for nmsl in new_models_type_selectors_list:
                        new_model_type_filenames += modules.config.get_model_filenames(modules.config.modelfile_path,
                                                                                       name_filter=nmsl)

                    new_model_type_filenames = sorted(set(new_model_type_filenames), key=new_model_type_filenames.index)

                    if y in new_model_type_filenames:
                        _value = y
                    else:
                        _value = new_model_type_filenames[0]

                    if "SDXL" in x:
                        return [gr.update(label=x, choices=new_model_type_filenames, value=_value),
                                gr.update(visible=True), gr.update(visible=True)]
                    if "HunyuanDiT" in x or "Flux" in x:
                        return [gr.update(label=x, choices=new_model_type_filenames, value=_value),
                                gr.update(visible=False, value="None"), gr.update(value="ALL_UnChecked")]
                    else:
                        return [gr.update(label=x, choices=new_model_type_filenames, value=_value),
                                gr.update(visible=False, value="None"), gr.update(visible=True)]


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
                    model_refresh = gr.Button(label='Refresh', value='\U0001f504 Refresh All Files',
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


        def adjust_refiner_model_config(x, y, z):
            r = modules.config.get_config_from_model_preset(y).get("default_refiner")
            if x == "Custom":
                refiner = z
            else:
                refiner = r
            print(f"x,y,refiner = {x} - {y} - {refiner}")
            if x.lower() == constants.TYPE_LIGHTNING:
                cis = ['None'] + [c for c in modules.config.model_filenames if constants.TYPE_LIGHTNING in c.lower()]
                return gr.update(label=x.title() + " model(for SDXL)", choices=cis, value=cis[0], show_label=True)
            elif x.lower() in [constants.TYPE_LCM, constants.TYPE_TURBO, constants.TYPE_HYPER_SD]:
                cis = ['None'] + [c for c in modules.config.model_filenames if x.lower() in c.lower()]
                return gr.update(label=x.title() + " model(for SDXL)", choices=cis, value=cis[0], show_label=True)
            else:
                return gr.update(label='Refiner (SDXL or SD 1.5)',
                                 choices=['None'] + modules.config.model_filenames,
                                 value=refiner, show_label=True)


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
                results += [m]
            else:
                results += ["Not Exist!->" + m]

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
                if r in ["", "None", "Not Exist!->"]:
                    results += ["None"]
                else:
                    results += ["Not Exist!->" + r]

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

            print(x)

            if x[0] not in modules.config.preset_filenames:
                results += [gr.update(choices=modules.config.preset_filenames, value="Not Exist!->")]
            else:
                results += [gr.update(choices=modules.config.preset_filenames)]

            selected_model_filenames = modules.config.get_model_filenames(modules.config.modelfile_path,
                                                                          name_filter=x[1])
            selected_model_filenames = sorted(set(selected_model_filenames), key=selected_model_filenames.index)

            results += [gr.update(value=x[1])]

            if x[2] not in modules.config.model_filenames:
                results += [gr.update(choices=['None'] + modules.config.model_filenames, value="Not Exist!->")]
            else:
                results += [gr.update(choices=['None'] + selected_model_filenames)]

            if x[3] not in [modules.config.model_filenames, "None"]:
                results += [gr.update(choices=['None'] + modules.config.model_filenames, value="Not Exist!->")]
            else:
                results += [gr.update(choices=['None'] + modules.config.model_filenames)]

            y = list(x[4:-2])
            z = [y[nn:nn + 3] for nn in range(0, len(y), 3)]
            print(z)
            for lf in z:
                if lf[1] is not None and "\\" in lf[1]:
                    lf[1].replace('\\\\', '\\')
                if lf[1] not in modules.config.lora_filenames or lf[1] is None:
                    results += [gr.update(value=False),
                                gr.update(choices=['None'] + modules.config.lora_filenames, value="Not Exist!->"),
                                gr.update(value=lf[2])]
                else:
                    results += [gr.update(value=True),
                                gr.update(choices=['None'] + modules.config.lora_filenames, value=lf[1]),
                                gr.update(value=lf[2])]

            if x[-2] not in modules.config.controlnet_lora_canny_filenames:
                results += [
                    gr.update(choices=['None'] + modules.config.controlnet_lora_canny_filenames,
                              value="Not Exist!->")]
            else:
                results += [gr.update(choices=['None'] + modules.config.controlnet_lora_canny_filenames)]

            if x[-1] not in modules.config.controlnet_lora_depth_filenames:
                results += [
                    gr.update(choices=['None'] + modules.config.controlnet_lora_depth_filenames,
                              value="Not Exist!->")]
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
            .then(fn=lambda: None, _js='refresh_grid_delayed', queue=False, show_progress=False)


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
            .then(fn=lambda: None, _js='playNotification').then(fn=lambda: None, _js='refresh_grid_delayed')

        # for notification_file in ['notification.ogg', 'notification.mp3']:
        #     if os.path.exists(notification_file):
        #         gr.Audio(interactive=False, value=notification_file, elem_id='audio_notification', visible=False)
        #         break

# dump_default_english_config()
app = gr.mount_gradio_app(app, shared.gradio_root.queue(concurrency_count=2, max_size=2), '/')
async_gradio_app = shared.gradio_root
async_gradio_app.launch(
    inbrowser=adapter.args_manager.args.in_browser,
    server_name=adapter.args_manager.args.listen,
    server_port=adapter.args_manager.args.port,
    share=adapter.args_manager.args.share,
    auth=check_auth if (adapter.args_manager.args.share or adapter.args_manager.args.listen) and auth_enabled else None,
    allowed_paths=[modules.config.path_outputs],
    blocked_paths=[constants.AUTH_FILENAME]
)
