import ctypes
import traceback

import time
import os
from multiprocessing import Process, Manager
import hashlib
import modules
from procedure.worker import taskManager

class Middle(taskManager):
    """
    """

    def __init__(self):
        super().__init__()
        self.old_md5 = ''
        self.b_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    def get_file_md5(self, file_path):
        with open(file_path, 'rb') as f:
            bytes = f.read()
            hash = hashlib.md5(bytes).hexdigest()
            # print(f'        <get_file_md5> [{file_path}]  | hash: [{hash}]')
            f.close()
        return hash

    def get_buffer(self, file_path, old_md5=None, b_path=None):
        blank_1 = " " * 7 + "<Get_Buffer>"
        blank_2 = " " * 20
        if not file_path:
            config_file_name = "current.api.json"
            file_path = os.path.join(b_path, "configs", config_file_name)
        new_md5 = self.get_file_md5(file_path)
        print(f'{blank_1}:{file_path}]')
        if new_md5 != old_md5:
            print(f'{blank_1}:{self.buffer} updated! \n{blank_2}NEW_MD5:<{new_md5}> different from OLD_MD5:<{old_md5}>')
            self.old_md5 = new_md5
            return file_path
        else:
            print(f'{blank_1}:{self.buffer} unchanged! \n{blank_2}NEW_MD5 same as OLD_MD5:<{old_md5}>')
            return

    def action(self, buffer, m_flag, file_path=None):
        blank_1 = "<MiddleMan>"
        blank_2 = " " * 20
        finished = False
        time_interval = 60
        while not finished:
            self.m_flag = m_flag.value
            # print(f'self.m_flag: {self.m_flag}')
            if not self.m_flag:
                print(f'{blank_1}\n{blank_2}Waiting {time_interval} seconds, check {buffer} again...')
                time.sleep(time_interval)
                b = self.get_buffer(file_path, old_md5=self.old_md5, b_path=self.b_path)
                if b:
                    buffer.append(b)

    def set_task_flag_file(self, flag="False"):
        flag_file = os.path.join(self.b_path, "configs", "TASK_ALLOW_FLAG")
        try:
            with open(flag_file, "w") as f:
                f.write(flag)
        finally:
            f.close()
        return flag

    def process(self, buffer, m_flag, first_run=True):
        blank_1 = "<WorkShop>"
        blank_2 = " " * 20
        if first_run:
            modules.default_pipeline.refresh_everything(
                refiner_model_name=modules.config.default_refiner_model_name,
                base_model_name=modules.config.default_base_model_name,
                loras=modules.config.default_loras
            )
            first_run = False
        time_interval = 45
        while True:
            self.async_tasks = list(buffer[:])
            print(f'{blank_1}\n{blank_2}Waiting {time_interval} seconds, inqury buffer-->{self.buffer} again...')
            time.sleep(time_interval)
            self.set_task_flag_file(flag="True")
            if len(self.async_tasks) > 0:
                self.set_task_flag_file(flag="False")
                print(f"{blank_1}\n{blank_2}Buffer:" + self.buffer.__str__())
                fp = self.async_tasks.pop()
                buffer[:] = []
                m_flag.value = True
                print(f"{blank_2}fp:" + fp.__str__())
                self.execution_start_time = time.perf_counter()
                try:
                    def import_config_key():
                        config = self.get_config_from_file(file=fp)
                        self.get_config_key(config=config)

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

                except:
                    traceback.print_exc()
                    m_flag.value = False
                finally:
                    self.build_image_wall(fp)
                    modules.default_pipeline.prepare_text_encoder(async_call=True)
                m_flag.value = False


if __name__ == '__main__':
    m = Middle()
    manager = Manager()
    buffer = manager.list()
    m_flag = manager.Value(ctypes.c_bool, False)
    p = Process(target=m.action, args=(buffer, m_flag))
    q = Process(target=m.process, args=(buffer, m_flag))

    tasks = [p, q]
    for i in tasks:
        i.start()

    p.join()
    q.join()
