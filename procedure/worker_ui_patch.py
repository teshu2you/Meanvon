import sys

from adapter.parameters import ImageGenerationParams
from util.printf import printF, MasterName
import procedure.worker as worker
import shared
import time
import modules.default_pipeline as pipeline
import traceback
import modules
import threading
import gradio as gr
import adapter.args_manager
from modules.auth import auth_enabled, check_auth
import modules.constants as constants
from adapter.task_queue import QueueTask, TaskQueue, TaskOutputs

task_manager = worker.taskManager(request_source="webui")
async_tasks = task_manager.async_tasks

try:
    flag = ""
    async_gradio_app = shared.gradio_root
    if async_gradio_app is not None:
        flag = f'''App started successful. Use the app with {str(async_gradio_app.local_url)} or {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}'''

    if hasattr(async_gradio_app, 'share') and async_gradio_app.share:
        flag += f''' or {async_gradio_app.share_url}'''
    print(flag)
except Exception as e:
    print(e)


def worker_ui():
    from procedure.worker import worker_queue, process_top, blocking_get_task_result
    from adapter.args import args
    from adapter.task_queue import TaskQueue
    worker_queue = TaskQueue(queue_size=args.queue_size, hisotry_size=args.queue_history,
                                    webhook_url=args.webhook_url, persistent=args.persistent)

    printF(name=MasterName.get_master_name(),
           info="[MeaVon-API] Task queue size: " + str(args.queue_size) +
                ", queue history size: " + str(args.queue_history) + ", webhook url: " + str(
               args.webhook_url)).printf()

    igp = ImageGenerationParams()

    while True:
        time.sleep(5)
        if len(async_tasks) > 0:
            current_task = async_tasks.pop(0)
            printF(name=MasterName.get_master_name(),
                   info="WebUI patch init current_task: {}".format(current_task.__dict__)).printf()
            try:
                worker_queue.add_task(type=adapter.task_queue.TaskType.text_2_img, req_param=igp)
                queue_worker = QueueTask(job_id=worker_queue.last_job_id, type=adapter.task_queue.TaskType.text_2_img,
                                       req_param=igp,
                                       in_queue_millis=int(round(time.time() * 1000)))
                task_manager.process_generate(current_task, worker_queue, queue_worker)
                if current_task:
                    task_manager.build_image_wall(current_task)
                current_task.yields.append(['finish', current_task.results])
                pipeline.prepare_text_encoder(async_call=True)
            except:
                traceback.print_exc()
                current_task.yields.append(['finish', current_task.results])
            finally:
                if task_manager.pid in modules.patch.patch_settings:
                    del modules.patch.patch_settings[task_manager.pid]
    pass


threading.Thread(target=worker_ui, daemon=True).start()
