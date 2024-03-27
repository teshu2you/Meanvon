import os
import time
import json
import shutil
import argparse

# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler
# class MyHandler(FileSystemEventHandler):
#     def on_any_event(self, event):
#         pass

# event_handler = MyHandler()
# observer = Observer()  # 定义监控类,多线程类 thread class
# observer.schedule(event_handler, './monitor_folder_1', recursive=True)  # 指定监控路径/触发对应的监控事件类
# observer.start()  # 将observer运行在同一个线程之内,不阻塞主进程运行,可以调度observer来停止该线程
# try:
#     while True:
#         time.sleep(1)  # 监控频率（1s1次，根据自己的需求进行监控）
# except KeyboardInterrupt:
#     observer.stop()
#     observer.join()

path = os.path.dirname(os.path.realpath(__file__))
current_config_file = "current.api.json"
backup_config_file = "backup.api.json"

class P:
    global path, current_config_file, backup_config_file, args

    @staticmethod
    def save_to_file(flag='file', s_file=None, s_args=None):
        t_file = os.path.join(path, "configs", current_config_file)
        print(f't_file: {t_file}')
        if 'file' == flag and s_file:
            s_file = os.path.join(path, "configs", s_file)
            print(f's_file: {s_file}')
            shutil.copyfile(s_file, t_file)
        elif 'args_doc' == flag and s_args:
            with open(t_file, "w") as f:
                f.write(s_args)
                f.close()
        else:
            raise Exception("Lack of parameters of file or args!")

    @staticmethod
    def backup_file():
        s_file = os.path.join(path, "configs", current_config_file)
        t_file = os.path.join(path, "configs", backup_config_file)
        shutil.copyfile(s_file, t_file)


if __name__ == '__main__':
    p = P()
    config_file = "base.api.json"
    parser = argparse.ArgumentParser(description="use fooocus by api")
    parser.add_argument('-b', '--backup', type=str, help="backup <current.api.json> to <backup.api.json>")
    parser.add_argument('-c', '--config', type=str, default=config_file, help="load <xxxxxxx.api.json> to <current.api.json>")
    parser.add_argument('-s', '--save', type=str, help="save   <config.api.json>  to <current.api.json>")
    parser.add_argument('-f', '--flag', type=str, choices=['file', 'args_doc'], help="set the source of args, file or args inputed")
    args = parser.parse_args()
    # s.backup_file()
    p.save_to_file(flag='file', s_file=args.config)

