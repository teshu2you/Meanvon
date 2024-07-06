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
        t_file = os.path.join(self.real_path, "configs", file)
        if not t_file:
            t_file = os.path.join(self.real_path, "configs", self.current_config_file)
        print(f'load_config_from_file ->t_file: {t_file}')
        with open(t_file, "r+", encoding="UTF-8") as f:
            data = json.load(f)
            f.close()
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


if __name__ == '__main__':
    p = TaskManager()
    p.create_table()
    p.start_connect_db()
    config_file = "current.api.json"
    main_body = ""
    rep_primary_list = ['kindly expression, amiable,warm,benevolent countenance,',
                        'blubbering expression, cry loudly,burst into tears,emotional breakdown,open mouth,tears on face',
                        'weep expression, tears silently, cry quietly,tears in face,',
                        'hysteric expression, crazy, shouting and screaming, ',
                        'coquettish expression, tease, tantalize,licking lips,',
                        'excited expression,tears brimming in eyes, eyes glistening with tears,eyes brimming over warm tears,tears welling up in eyes,',
                        'orgasm expression, N-VAR,sexual excitement ,climaxs,open mouth,shy,hot and red face, ',
                        'grievance expression,aggrieved,feel wronged,be misunderstood,suffer from injustice,',
                        'heroic expression, bold and generous, enthusiasm ,zeal ,warmth, devotion,zest ,vitality , fervent,ardour , vigour , pep,heartily, uproarious,',
                        'avaricious expression, greedy,rapacious,',
                        'contemptuous expression, glaring,rolling eyes ',
                        'fantasizing expression, daydreaming, imaginative, creative,',
                        'affectionate expression, loving, tender, caring,',
                        'happiness expression, satisfied, well-being , bliss , felicity , intoxicated ',
                        'serious expression, brave, resolute, passionate, ',
                        'disappointed expression, disheartened, disillusioned, discouraged,',
                        'empathetic expression, understanding, compassionate, caring,',
                        'insecure expression, uncertain, uneasy, unconfident,',
                        'nostalgic expression, wistful, yearning, sentimental,',
                        'irritated expression, annoyed, vexed, exasperated,',
                        'compassionate expression, empathetic, caring, kind,',
                        'confident expression, self-assured, poised, assertive,',
                        'envious expression, jealous, covetous, resentful,',
                        'mischievous expression, impish, sly, playful,',
                        'doubtful expression, uncertain, hesitant, unsure,',
                        'relieved expression, reassured, comforted, soothed,',
                        'overwhelmed expression, swamped, flooded, inundated,',
                        'ecstatic expression, jubilant, overjoyed, exuberant,',
                        'exhausted expression, worn out, fatigued, drained,',
                        'flattered expression, gratified, pleased, complimented,',
                        'humble expression, modest, unassuming, unpretentious,',
                        'guilty expression, ashamed, regretful, remorseful',
                        'regretful expression, remorseful, repentant, rueful,',
                        'shocked expression, stunned, astonished, astounded',
                        'apologetic expression, remorseful, regretful, sorry,',
                        'concerned expression, troubled, uneasy, fretful,',
                        'smug expression, conceited, self-satisfied, arrogant,',
                        'coy expression, flirtatious, playful, teasing,',
                        'daydreaming expression, lost in thought, contemplative, pensive,',
                        'overjoyed expression, ecstatic, elated, thrilled,',
                        'nervous expression, tense, jumpy, uneasy',
                        'frustrated expression, agitated, exasperated, annoyed,',
                        'sleepy expression, drowsy, tired, fatigued,',
                        'focused expression, concentrated, absorbed, engaged,',
                        'thoughtful expression, pensive, reflective, contemplative,',
                        'anxious expression, worried, uneasy, apprehensive,',
                        'excited expression, thrilled, eager, enthusiastic,',
                        'overjoyed expression, ecstatic, elated, thrilled,',
                        'determined expression, resolute, purposeful, firm,',
                        'content expression, satisfied, pleased, gratified,',
                        'embarrassed expression, bashful, shy, blushing,',
                        'neutral expression, impassive, stoic, indifferent,',
                        'amused expression, entertained, tickled, delighted,',
                        'curious expression, inquisitive, intrigued, interested,',
                        'skeptical expression, doubtful, questioning, suspicious,',
                        'proud expression, confident, self-assured, triumphant,',
                        'disgusted expression, repulsed, revolted, nauseated,',
                        'surprised expression, astonished, amazed, startled,',
                        'angry expression, furious, irritated, annoyed,',
                        'confused expression, puzzled, perplexed, bewildered,',
                        'fearful expression, afraid, terrified, panicked,',
                        'sad expression, sorrowful, downcast, melancholic,',
                        'mischievous expression, impish, sly, playful,'
                        ]

    for idx, val in enumerate(rep_primary_list):
        print(f'body is: {idx} : {val}')
        main_body = val
        rep_dict = {
            "keep_input_names": "true",
            "input_image_filename": main_body,
            "aspect_ratios_selection": "1024×1024 (1:1)",
            "image_number": 2,
            "sharpness": 2.0,
            "sampler_name": "dpmpp_2m_sde_gpu",
            "scheduler_name": "karras",
            "custom_steps": 30,
            "custom_switch": 0.5,
            "cfg": 4,
            "prompt": main_body + ",raw photo, realistic, 1girl, 18 years old girl, (extreme detailed face, detailed skin), whole body, shirt,",
            "negative_prompt": "ng_deepnegative_v1_75t, badhandv4, (worst quality:2), (low quality:2), (normal quality:2), lowres, bad anatomy, bad hands, normal quality, ((monochrome)), ((grayscale)), (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, (worst quality, low quality:2), jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, BadDream,",
            "style_selections": ['Fooocus V2', 'Fooocus Enhance', 'Fooocus Photograph'],
            "base_model_name": "JuggernautXL_X_X_RunDiffusion.safetensors",
            "refiner_model_name": "",
            "loras": [['sd_xl_offset_example-lora_1.0.safetensors', 0.2], ['SDXL_LORA_艺术_more_art-full_v1.safetensors', 0.7], ['SDXL_LORA_控制_add-detail-xl增加细节.safetensors', 0.8],
                      ['None', 0.47], ['None', 1.0]]
        }
        try:
            p.process(f=config_file, s=val, **rep_dict)
        except Exception as e:
            print(f"{e}")
            break

    p.close_db()
