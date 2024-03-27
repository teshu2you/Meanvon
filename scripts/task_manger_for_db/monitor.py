import ctypes
import datetime
import traceback
import json
from workshop import Api
import time
import os
import modules
import sqlite3
from common import Common


class Middle(Common, Api):
    """
    """

    def __init__(self):
        super().__init__()
        Api.__init__(self)
        self.table_name = "Task"

    def query_all(self):
        return self.query(table=self.table_name, data={"*": ""})

    def delete_all(self):
        return self.delete(table=self.table_name, data={"*": ""})

    def query_one(self, data):
        return self.query(table=self.table_name, data=data)

    def delete_one(self, data):
        return self.delete(table=self.table_name, data=data)

    def update_one(self, data):
        return self.update(table=self.table_name, data=data)

    def make_dict(self, task_item):
        _dict = {}
        _dict_keys = ["prompt",
                      "negative_prompt",
                      "style_selections",
                      "performance_selection",
                      "aspect_ratios_selection",
                      "image_number",
                      "image_seed",
                      "sharpness",
                      "sampler_name",
                      "scheduler_name",
                      "custom_steps",
                      "custom_switch",
                      "cfg",
                      "base_model_name",
                      "refiner_model_name",
                      "base_clip_skip",
                      "refiner_clip_skip",
                      "loras",
                      "save_metadata_json",
                      "save_metadata_image",
                      "img2img_mode",
                      "img2img_start_step",
                      "img2img_denoise",
                      "img2img_scale",
                      "revision_mode",
                      "positive_prompt_strength",
                      "negative_prompt_strength",
                      "revision_strength_1",
                      "revision_strength_2",
                      "revision_strength_3",
                      "revision_strength_4",
                      "same_seed_for_all",
                      "output_format",
                      "control_lora_canny",
                      "canny_edge_low",
                      "canny_edge_high",
                      "canny_start",
                      "canny_stop",
                      "canny_strength",
                      "canny_model",
                      "control_lora_depth",
                      "depth_start",
                      "depth_stop",
                      "depth_strength",
                      "depth_model",
                      "input_image_checkbox",
                      "current_tab",
                      "uov_method",
                      "refiner_switch",
                      "outpaint_selections",
                      "cn_tasks",
                      "input_gallery",
                      "revision_gallery",
                      "keep_input_names",
                      "freeu_enabled",
                      "loras_raw",
                      "freeu_b1",
                      "freeu_b2",
                      "freeu_s1",
                      "freeu_s2",
                      "raw_style_selections",
                      "use_expansion",
                      "revision_gallery_size",
                      "initial_latent",
                      "denoising_strength",
                      "tiled",
                      "skip_prompt_processing",
                      "refiner_swap_method",
                      "raw_prompt",
                      "raw_negative_prompt",
                      "inpaint_image",
                      "inpaint_mask",
                      "inpaint_head_model_path",
                      "controlnet_canny_path",
                      "controlnet_cpds_path",
                      "clip_vision_path",
                      "ip_negative_path",
                      "ip_adapter_path",
                      "goals",
                      "tasks",
                      "steps",
                      "switch",
                      "mixing_image_prompt_and_vary_upscale",
                      "overwrite_step",
                      "overwrite_switch",
                      "overwrite_width",
                      "overwrite_height",
                      "width",
                      "height",
                      "results",
                      "metadata_strings",
                      "input_gallery_size",
                      "cfg_scale",
                      "imgs",
                      "use_style",
                      "input_image_path",
                      "adaptive_cfg",
                      "adm_scaler_positive",
                      "adm_scaler_negative",
                      "adm_scaler_end",
                      "mixing_image_prompt_and_inpaint",
                      "inpaint_engine",
                      "overwrite_vary_strength",
                      "start_step",
                      "denoise",
                      "input_image_filename",
                      "debugging_cn_preprocessor",
                      "revision_images_filenames",
                      "overwrite_upscale_strength",
                      "generate_image_grid",
                      "use_synthetic_refiner"]
        _dict_vals = task_item[2:]
        _dict_vals_new = []
        for jj in _dict_vals:
            # print(jj)
            if isinstance(jj, str):
                if "[" in jj:
                    jj = jj.replace("'", '"')
                    jj = json.loads(jj)
                if jj in ['none', 'None', "NONE"] and isinstance(jj, str):
                    jj = None

                if jj in ["False", "false", "FALSE"]:
                    jj = False

                if jj in ["True", "true", "TRUE"]:
                    jj = True

            _dict_vals_new.append(jj)

        _dict = dict(zip(_dict_keys, _dict_vals_new))
        print(_dict)
        return _dict

    def process(self, first_run=True):
        blank_1 = "<WorkShop>"
        blank_2 = " " * 20
        if first_run:
            modules.default_pipeline.refresh_everything(
                refiner_model_name=modules.config.default_refiner_model_name,
                base_model_name=modules.config.default_base_model_name,
                loras=modules.config.default_loras
            )
            first_run = False

        num = 0
        while True:
            self.async_tasks = self.query_all()
            total_num = len(self.async_tasks)
            print(f'{blank_1}\n{blank_2} total_num:{total_num} vs num:{num}...')
            if total_num != num:
                num = 0
                for kk in self.async_tasks:
                    num += 1
                    print(f'{blank_1}\n{blank_2} {kk}...')
                    if kk[1] != self.STATUS.VALID:
                        print(f'{blank_1}\n{blank_2} ID:{kk[0]} --- [{kk[1]}] --- Ignore!')
                        continue
                    print(f'{blank_1}\n{blank_2} ID:{kk[0]} --- [{kk[1]}] --- Ready to generate...')
                    self.execution_start_time = time.perf_counter()
                    try:
                        def import_config_key():
                            self.get_config_key(config=self.make_dict(task_item=kk))

                        n = 1
                        for x in [import_config_key,
                                  self.pre_process,
                                  self.download_image_func_models,
                                  self.encode_prompts,
                                  self.manage_cns,
                                  self.get_advanced_parameters,
                                  self.check_vary_in_goals,
                                  self.check_upscale_in_goals,
                                  self.check_inpaint_in_goals,
                                  self.check_cn_in_goals,
                                  self.generate_images,
                                  self.post_process
                                  ]:
                            print('-' * 200)
                            start_time = time.perf_counter()
                            print(f'Step {n}: {x.__name__}')
                            x()
                            cost_time = time.perf_counter() - start_time
                            print(
                                f'\n                          Cost Time: <<<<<<<<<<<<<< {cost_time:.2f} seconds >>>>>>>>>>>>>>>>>')
                            n += 1
                        self.update_one(data={kk[0]: ["task_status", self.STATUS.INVALID]})
                    except:
                        traceback.print_exc()
                    finally:
                        self.build_image_wall(kk)
                        modules.default_pipeline.prepare_text_encoder(async_call=True)
            else:
                now = datetime.datetime.now()
                print(f'{blank_1}\n{blank_2} No Tasks, waiting 60s!  ---  Now: {now}')
                time.sleep(60)


if __name__ == '__main__':
    m = Middle()
    print(m.query_all())
    # m.delete_all()
    m.process()
