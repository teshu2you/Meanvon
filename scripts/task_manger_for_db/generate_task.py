import os
import time
import json
import sqlite3
from common import Common


class TaskManager(Common):
    def __init__(self):
        super().__init__()
        self.task_id = None
        self.task_status = self.STATUS.VALID
        self.table_name = "Task"
        self.real_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.current_config_file = "current.api.json"

    def load_config_from_file(self, file=None):
        t_file = os.path.join(self.real_path, "../task_configs", file)
        if not t_file:
            t_file = os.path.join(self.real_path, "../task_configs", self.current_config_file)
        print(f'load_config_from_file ->t_file: {t_file}')

        if t_file.endswith('.json'):
            with open(t_file, "r+", encoding="UTF-8") as f:
                data = json.load(f)
        else:
            with open(t_file, "r+", encoding="UTF-8") as f:
                lines = f.readlines()
                # 过滤掉行首包含 '#' 的行和空行
                filtered_lines = [line.strip() for line in lines if not line.strip().startswith('#') and line.strip()]
                data = filtered_lines

        return data

    def process(self, f=None, s="", **rep_dict):
        print(f'<<<<{s}>>>> save to db ...')
        self.save_to_db(file=f, **rep_dict)

    def save_to_db(self, file=None, **key):
        data = self.load_config_from_file(file=file)
        data.update(**key)
        data.update({
            "task_status": self.STATUS.VALID
        })
        self.insert(table=self.table_name, data=data)

    def get_content_from_file(self, file=None):
        data = self.load_config_from_file(file=file)
        return data


if __name__ == '__main__':
    p = TaskManager()
    p.create_table()
    p.start_connect_db()
    config_file = "current.api.json"
    main_body = ""

    rep_model_list = p.get_content_from_file(file="models_list.txt")
    rep_primary_list = p.get_content_from_file(file="prompts_list.txt")

    for idx_m, val_m in enumerate(rep_model_list):
        print(f'model is: {idx_m} : {val_m}')
        for idx_p, val_p in enumerate(rep_primary_list):
            print(f'body is: {idx_p} : {val_p}')
            main_body = val_p
            rep_dict = {
                "keep_input_names": "true",
                "input_image_filename": main_body,
                "aspect_ratios_selection": "1024×1024 (1:1)",
                "image_number": 3,
                "sharpness": 2.0,
                "sampler_name": "dpmpp_2m_sde_gpu",
                "scheduler_name": "karras",
                "custom_steps": 30,
                "custom_switch": 0.5,
                "cfg": 5,
                "prompt": main_body + "",
                "negative_prompt": "paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), dot, mole, lowres, normal quality, monochrome, grayscale, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature,",
                "style_selections": [],
                "base_model_name": val_m,
                "refiner_model_name": "",
                "loras": [['true', 'Primary\SDXL_LORA_控制_add-detail-xl增加细节.safetensors', 0.97],
                          ['true', 'Primary\SDXL_LORA_FILM_PHOTOGRAPHY_STYLE_BetaV0.4.safetensors', 0.8],
                          ['true', '人物\SDXL_LORA_人物_(dili)_1.0SDXL_1.safetensors', 0.8],
                          ['false', 'None', 1.0], ['false', 'None', 1.0]]
            }
            try:
                p.process(f=config_file, s=val_p, **rep_dict)
            except Exception as e:
                print(f"{e}")
                break

    p.close_db()
