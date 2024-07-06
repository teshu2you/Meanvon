import os
import time
import json


class P:
    def __init__(self):
        self.real_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.current_config_file = "current.api.json"

    def load_config_from_file(self, file=None):
        t_file = os.path.join(self.real_path, "configs", file)
        if not t_file:
            t_file = os.path.join(self.real_path, "configs", self.current_config_file)
        print(f'load_config_from_file ->t_file: {t_file}')
        with open(t_file, "r+", encoding="UTF-8") as f:
            data = json.load(f)
            f.close()
        return data

    def dump_config_to_file(self, file=None, args=None):
        t_file = os.path.join(self.real_path, "configs", file)
        if not t_file:
            t_file = os.path.join(self.real_path, "configs", self.current_config_file)
        print(f'dump_config_to_file ->t_file: {t_file}')
        with open(t_file, "w") as f:
            json.dump(args, f)
            f.close()

    def replace_content(self, file=None, **key):
        t_file = os.path.join(self.real_path, "configs", file)
        if not t_file:
            t_file = os.path.join(self.real_path, "configs", self.current_config_file)
        print(f'replace_content ->t_file: {t_file}')
        data = self.load_config_from_file(file=file)
        data.update(**key)
        self.dump_config_to_file(file=file, args=data)

    def check_task_flag(self):
        flag_file = os.path.join(self.real_path, "configs", "TASK_ALLOW_FLAG")
        flag = ""
        try:
            with open(flag_file, "r+") as f:
                flag = f.readline()
        except FileNotFoundError:
            with open(flag_file, "w") as f:
                f.write("")
        finally:
            f.close()
        return flag

    def set_task_flag_file(self, flag="False"):
        flag_file = os.path.join(self.real_path, "configs", "TASK_ALLOW_FLAG")
        try:
            with open(flag_file, "w") as f:
                f.write(flag)
        finally:
            f.close()
        return flag

    def process(self, f=None, s="", t=60, **rep_dict):
        print(f'check task flag: {self.check_task_flag()} , <<<<{s}>>>> starting to replace ...')
        p.replace_content(file=f, **rep_dict)
        print(f'sleep {t} seconds ...')
        time.sleep(t)


if __name__ == '__main__':
    p = P()
    config_file = "current.api.json"
    main_body = ""
    rep_primary_list = [
         'Mk Mosaic', 'Mk Van Gogh', 'Mk Coloring Book', 'Mk Singer Sargent', 'Mk Pollock',
         'Mk Basquiat', 'Mk Andy Warhol', 'Mk Halftone Print', 'Mk Gond Painting', 'Mk Albumen Print',
         'Mk Aquatint Print', 'Mk Anthotype Print', 'Mk Inuit Carving', 'Mk Bromoil Print', 'Mk Calotype Print',
         'Mk Color Sketchnote', 'Mk Cibulak Porcelain', 'Mk Alcohol Ink Art', 'Mk One Line Art', 'Mk Blacklight Paint',
         'Mk Carnival Glass', 'Mk Cyanotype Print', 'Mk Cross Stitching', 'Mk Encaustic Paint', 'Mk Embroidery',
         'Mk Gyotaku', 'Mk Luminogram', 'Mk Lite Brite Art', 'Mk Mokume Gane', 'Pebble Art', 'Mk Palekh',
         'Mk Suminagashi', 'Mk Scrimshaw', 'Mk Shibori', 'Mk Vitreous Enamel', 'Mk Ukiyo E',
         'Mk Vintage Airline Poster', 'Mk Vintage Travel Poster', 'Mk Bauhaus Style', 'Mk Afrofuturism', 'Mk Atompunk',
         'Mk Constructivism', 'Mk Chicano Art', 'Mk De Stijl', 'Mk Dayak Art', 'Mk Fayum Portrait',
         'Mk Illuminated Manuscript', 'Mk Kalighat Painting', 'Mk Madhubani Painting', 'Mk Pictorialism',
         'Mk Pichwai Painting', 'Mk Patachitra Painting', 'Mk Samoan Art Inspired', 'Mk Tlingit Art', 'Mk Adnate Style',
         'Mk Ron English Style', 'Mk Shepard Fairey Style'
    ]
    for idx, val in enumerate(rep_primary_list):
        print(f'body is: {idx} : {val}')
        main_body = val
        rep_dict = {
            "input_image_filename": main_body,
            "aspect_ratios_selection": "1024×1024 (1:1)",
            "image_number": 1,
            "prompt": "1girl",
            "negative_prompt": "(low quality:2),ng_deepnegative_v1_75t,badhandv4,(holding:2.2),(worst quality:2),(low quality:2),(normal quality:2),lowres,bad anatomy,bad hands,normal quality,((monochrome)),((grayscale)),",
            "style_selections": ['Fooocus V2'] + [main_body],
            "base_model_name": "JuggernautXL_X_X_RunDiffusion.safetensors",
            "refiner_model_name": "None",
            "loras": [['sd_xl_offset_example-lora_1.0.safetensors', 0.1], ['None', 0.45], ['None', 0.35],
                      ['None', 0.47], ['None', 1.0]]
        }

        # p.set_task_flag_file(flag="True")
        # p.process(f=config_file, s=val, t=60, **rep_dict)

        waiting_flag = False
        while not waiting_flag:
            _time = 10
            print(f'Sleep {_time} seconds ...')
            time.sleep(_time)
            # task_flag True ---> can be add new task, False ---> task processing, not allowed to add new task
            task_flag = p.check_task_flag()
            print(f'Check task flag: {task_flag} , <<<<{val}>>>> waiting ...')
            if "True" in task_flag:
                p.process(f=config_file, s=val, t=60, **rep_dict)
            task_flag = p.check_task_flag()
            if "True" in task_flag:
                waiting_flag = True
