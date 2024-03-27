import os
import time
import json
import sqlite3
from sqlite3 import OperationalError
import re
class Common:
    class STATUS:
        VALID = "valid"
        INVALID = "invalid"

    def __init__(self):
        self.real_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.task_db = os.path.join(self.real_path, "db", "task.db")
        self.conn = sqlite3.connect(self.task_db)
        self.data_storage = {}

    def create_table(self, table_name=None, sql=""):
        c = self.conn.cursor()
        if table_name is None:
            table_name = "Task"
        try:
            if sql == "":
                c.execute('''CREATE TABLE {} 
                        (id INTEGER PRIMARY   KEY     AUTOINCREMENT,
                        task_status        CHAR(10),  
                        prompt                  TEXT    ,
                        negative_prompt         TEXT    ,
                        style_selections        TEXT    ,
                        performance_selection    CHAR(10),
                        aspect_ratios_selection  CHAR(25),
                        image_number        INTEGER,  
                        image_seed          INTEGER,
                        sharpness           REAL,
                        sampler_name        CHAR(10),
                        scheduler_name      CHAR(10),
                        custom_steps        INTEGER,
                        custom_switch       REAL,
                        cfg                 REAL,
                        base_model_name     CHAR(100),
                        refiner_model_name  CHAR(100),
                        base_clip_skip     INTEGER,
                        refiner_clip_skip  INTEGER,
                        loras               TEXT    ,
                        save_metadata_json  CHAR(10),
                        save_metadata_image   CHAR(10),
                        img2img_mode          CHAR(10), 
                        img2img_start_step    REAL,
                        img2img_denoise       REAL,
                        img2img_scale         REAL,
                        revision_mode         CHAR(10),
                        positive_prompt_strength     REAL,
                        negative_prompt_strength     REAL,
                        revision_strength_1         REAL,
                        revision_strength_2         REAL,
                        revision_strength_3         REAL,
                        revision_strength_4         REAL,
                        same_seed_for_all                                       CHAR(10), 
                        output_format                                           CHAR(10), 
                        control_lora_canny                                      CHAR(10), 
                        canny_edge_low                                          REAL, 
                        canny_edge_high                                         REAL, 
                        canny_start                                             REAL, 
                        canny_stop                                              REAL, 
                        canny_strength                                          REAL, 
                        canny_model                                             CHAR(100), 
                        control_lora_depth                                      CHAR(10), 
                        depth_start                                             REAL, 
                        depth_stop                                              REAL, 
                        depth_strength                                          REAL, 
                        depth_model                                             CHAR(100), 
                        input_image_checkbox                                    CHAR(10), 
                        current_tab                                             CHAR(10), 
                        uov_method                                              CHAR(10), 
                        refiner_switch                                          REAL, 
                        outpaint_selections                                     CHAR(100), 
                        cn_tasks                                                CHAR(200), 
                        input_gallery                                           CHAR(10), 
                        revision_gallery                                        CHAR(10), 
                        keep_input_names                                        CHAR(10), 
                        freeu_enabled                                           CHAR(10), 
                        loras_raw                                               CHAR(10), 
                        freeu_b1                                                REAL, 
                        freeu_b2                                                REAL, 
                        freeu_s1                                                REAL, 
                        freeu_s2                                                REAL, 
                        raw_style_selections                                    CHAR(10), 
                        use_expansion                                           CHAR(10), 
                        revision_gallery_size                                   INTEGER, 
                        initial_latent                                          CHAR(10), 
                        denoising_strength                                      REAL, 
                        tiled                                                   CHAR(10), 
                        skip_prompt_processing                                  CHAR(10), 
                        refiner_swap_method                                     CHAR(10), 
                        raw_prompt                                              CHAR(10), 
                        raw_negative_prompt                                     CHAR(10), 
                        inpaint_image                                           CHAR(10), 
                        inpaint_mask                                            CHAR(10), 
                        inpaint_head_model_path                                 CHAR(10), 
                        controlnet_canny_path                                   CHAR(10), 
                        controlnet_cpds_path                                    CHAR(10), 
                        clip_vision_path                                        CHAR(10), 
                        ip_negative_path                                        CHAR(10), 
                        ip_adapter_path                                         CHAR(10), 
                        goals                                                   CHAR(10), 
                        tasks                                                   CHAR(10), 
                        steps                                                   INTEGER, 
                        switch                                                  INTEGER, 
                        mixing_image_prompt_and_vary_upscale                    CHAR(10), 
                        overwrite_step                                          INTEGER, 
                        overwrite_switch                                        INTEGER, 
                        overwrite_width                                         INTEGER, 
                        overwrite_height                                        INTEGER, 
                        width                                                   INTEGER, 
                        height                                                  INTEGER, 
                        results                                                 CHAR(1000), 
                        metadata_strings                                        CHAR(10), 
                        input_gallery_size                                      INTEGER, 
                        cfg_scale                                               REAL, 
                        imgs                                                    CHAR(100), 
                        use_style                                               CHAR(100), 
                        input_image_path                                        CHAR(100), 
                        adaptive_cfg                                            REAL, 
                        adm_scaler_positive                                     REAL, 
                        adm_scaler_negative                                     REAL, 
                        adm_scaler_end                                          REAL, 
                        mixing_image_prompt_and_inpaint                         CHAR(10), 
                        inpaint_engine                                          CHAR(10), 
                        overwrite_vary_strength                                 REAL, 
                        start_step                                              INTEGER, 
                        denoise                                                 REAL, 
                        input_image_filename                                    CHAR(200), 
                        debugging_cn_preprocessor                               CHAR(10), 
                        revision_images_filenames                               CHAR(100), 
                        overwrite_upscale_strength                              REAL, 
                        generate_image_grid                                     CHAR(10), 
                        use_synthetic_refiner                                   CHAR(10))'''.format(table_name))
            else:
                c.execute(sql)
            self.conn.commit()
            c.close()
            print("数据表创建成功")
        except OperationalError as o:
            print(f"OperationalError: {o}")
        except Exception as e:
            print(e)
        finally:
            self.conn.close()

    def start_connect_db(self):
        self.conn = sqlite3.connect(self.task_db)

    def close_db(self):
        self.conn.cursor().close()
        self.conn.close()

    def is_table_exited(self, table_name):
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT name from {} where type='table' order by name".format(table_name))
            if table_name in cur.fetchall():
                return True
            else:
                return False
        except sqlite3.Error as e:
            print(e)
            return False

    def insert(self, table: str, data: dict):
        if not data:
            return
        print("data:{}".format(data))
        key = []
        val = []
        for k, v in data.items():
            key.append(k)
            if isinstance(v, list) or isinstance(v, int) or isinstance(v, bool) or isinstance(v, float) or v is None:
                v = str(v)
            v = '"' + v + '"'
            val.append(v)
        print(f"key:{key}")
        print(f"val:{val}")
        key = ",".join(key)
        val = ",".join(val)

        sql = "INSERT INTO {} ({}) VALUES({})".format(
                table, key, val)
        print(f"sql:{sql}")
        r = self.execute(sql)
        self.conn.commit()
        print("insert succeeded!")
        if table in self.data_storage:
            self.data_storage[table].clear()
        return r

    def delete(self, table: str, data: dict):
        if not data:
            return
        for k, v in data.items():
            key = k
            val = v
        print("data:{}".format(data))
        if "*" in data.keys():
            sql = "DELETE FROM {}".format(table)
        else:
            sql = "DELETE FROM {} WHERE id='{}' AND {}".format(table, key, val)
        r = self.execute(sql)
        print("delete succeeded!")
        if table in self.data_storage:
            self.data_storage[table].clear()
        return r

    def update(self, table: str, data: dict):
        if not data:
            return
        for k, v in data.items():
            key = k
            val = v
        print("data:{}".format(data))
        sql = "UPDATE {} SET {}='{}' where id='{}'".format(table, val[0], val[1], key)
        r = self.execute(sql)
        print("update succeeded!")
        if table in self.data_storage:
            self.data_storage[table].clear()
        return r

    def query(self, table: str, data: dict):
        if not data:
            return
        for k, v in data.items():
            key = k
            val = v
        print("data:{}".format(data))
        if "*" in data.keys():
            sql = "SELECT * FROM {}".format(table)
        else:
            sql = "SELECT {} FROM ".format("{} where {}".format(key, val))
        r = self.execute(sql)
        print("query succeeded!")
        if table in self.data_storage:
            self.data_storage[table].clear()
        return r

    def execute(self, sql):
        try:
            r = self.conn.cursor().execute(sql)
            self.conn.commit()
            return r.fetchall()
        except Exception as e:
            print(f"execute failed! Error: {e}")
            self.conn.rollback()

    # 查询sql 没缓存
    def fetchall(self, sql) -> list:
        if self.conn is None:
            # 触发安装程序
            raise IOError('db')

        cur = self.conn.cursor()
        cur.execute(sql)
        print("fetchall succeeded!")
        return cur.fetchall()

    def non_empty(self, i: any) -> bool:
        return not self.empty(i)

    def empty(self, i: any) -> bool:
        if i is None:
            return True
        if isinstance(i, str):
            return i == ''
        if isinstance(i, list) or isinstance(i, tuple):
            return len(i) == 0
        if isinstance(i, dict):
            return i == {}
        if isinstance(i, int) or isinstance(i, float):
            return i == 0
        return False
    # 获取sql中的表名

    def get_table_name(self, sql):
        return list(set(re.findall("(av_[a-z]+)", sql)))

    def list_in_str(target_list: tuple, target_string: str) -> bool:
        for item in target_list:
            if item in target_string:
                return True
        return False

