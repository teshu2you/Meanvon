import argparse
import os
import re
import shutil
import subprocess
import sys
from importlib.util import find_spec
from threading import Thread
import time
import uvicorn
import contextlib
import threading
from fastapi import FastAPI
from contextlib import asynccontextmanager
from version import *
from boot.pre_check import PreCheck
from adapter.base_args import add_base_args
from adapter.args import args
from util.printf import printF, MasterName
from adapter.task_queue import QueueTask, TaskQueue, TaskOutputs
from modules.meta_parser import get_metadata_parser, MetadataScheme
from procedure.worker import task_schedule_loop

worker_queue: TaskQueue = None
queue_task: QueueTask = None
last_model_name = None

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TORCH_INDEX_URL"] = ""

python = sys.executable
default_command_live = True
index_url = os.environ.get('INDEX_URL', "")
re_requirement = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")

fooocus_name = 'Fooocus'
meanvon_name = 'MeanVon'
requirements_file = "requirements_versions.txt"

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = modules_path
pc = PreCheck(default_command_live=default_command_live, index_url=index_url, re_requirement=re_requirement,
              script_path=script_path, requirements_file=requirements_file)
printF(name="System ARGV", info=sys.argv).printf()
printF(name="Python version", info=sys.version).printf()
printF(name="MeanVon Main version", info=main_version).printf()
printF(name="MeanVon API version", info=api_version).printf()


class UvicornServer(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        # Start task schedule thread
        task_schedule_thread = Thread(target=task_schedule_loop(request_source="api"), daemon=True)
        task_schedule_thread.start()
        # try:
        #     while not self.started:
        #         time.sleep(1e-3)
        #     yield
        # finally:
        #     self.should_exit = True
        #     task_schedule_thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_base_args(parser, True)

    args, _ = parser.parse_known_args()
    printF(name="args", info=args).printf()
    printF(name="install dependents", info="torch & torchvision").printf()
    pc.install_dependents(args)

    from adapter.args import args

    # if pc.prepare_environments(args):
    #     sys.argv = [sys.argv[0]]
    #     # Start api server
    #     from adapter.api import start_app
    #
    #     print(f"args: {args}")
    #     config = start_app(args)
    #     UvicornServer(config=config).run_in_thread()

    if pc.prepare_environments(args):
        sys.argv = [sys.argv[0]]

        # task_schedule_thread = Thread(target=task_schedule_loop(request_source="api"), daemon=True)
        # task_schedule_thread.start()

        # Start api server
        from adapter.api import start_app

        # app_thread = Thread(target=start_app(args), daemon=True)
        # app_thread.start()

        print(f"args: {args}")
        config = start_app(args)


