import argparse
import os
import re
import shutil
import subprocess
import sys
from importlib.util import find_spec
from threading import Thread
from version import *
from util.printf import printF, MasterName
import modules.meta_parser

preset_content = {}
preset_prepared = modules.meta_parser.parse_meta_from_preset(preset_content)
default_model = preset_prepared.get('base_model')
previous_default_models = preset_prepared.get('previous_default_models', [])


class PreCheck:
    """
    pre0-check repo.
    """

    def __init__(self, default_command_live, index_url, re_requirement, script_path, requirements_file):
        self.name = 'MeanVon'
        self.default_command_live = default_command_live
        self.index_url = index_url
        self.re_requirement = re_requirement
        self.script_path = script_path
        self.requirements_file = requirements_file

    # This function was copied from [Fooocus](https://github.com/lllyasviel/Fooocus) repository.
    def valid_path(self, func, path):
        import stat
        if not os.access(path, os.W_OK):
            os.chmod(path, stat.S_IWUSR)
            func(path)
        else:
            raise 'Failed to invoke "shutil.rmtree", git management failed.'

    # This function was copied from [Fooocus](https://github.com/lllyasviel/Fooocus) repository.
    def run(self, command, desc=None, errdesc=None, custom_env=None) -> str:
        if desc is not None:
            print(desc)

        run_kwargs = {
            "args": command,
            "shell": True,
            "env": os.environ if custom_env is None else custom_env,
            "encoding": 'utf8',
            "errors": 'ignore',
        }

        if not self.default_command_live:
            run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

        result = subprocess.run(**run_kwargs)

        if result.returncode != 0:
            error_bits = [
                f"{errdesc or 'Error running command'}.",
                f"Command: {command}",
                f"Error code: {result.returncode}",
            ]
            if result.stdout:
                error_bits.append(f"stdout: {result.stdout}")
            if result.stderr:
                error_bits.append(f"stderr: {result.stderr}")
            raise RuntimeError("\n".join(error_bits))

        return result.stdout or ""

    # This function was copied from [Fooocus](https://github.com/lllyasviel/Fooocus) repository.
    def run_pip(self, command, desc=None):
        try:
            index_url_line = f' --index-url {self.index_url}' if self.index_url != '' else ''
            return self.run(f'"{sys.executable}" -m pip {command} --prefer-binary{index_url_line}',
                            desc=f"Installing {desc}",
                            errdesc=f"Couldn't install {desc}")
        except Exception as e:
            print(e)
            print(f'CMD Failed {desc}: {command}')
            return None

    # This function was copied from [Fooocus](https://github.com/lllyasviel/Fooocus) repository.
    def requirements_met(self, requirements_file):
        """
        Does a simple parse of a requirements.txt file to determine if all requirements in it
        are already installed. Returns True if so, False if not installed or parsing fails.
        """

        import importlib.metadata
        import packaging.version

        with open(requirements_file, "r", encoding="utf8") as file:
            for line in file:
                if line.strip() == "":
                    continue

                m = re.match(self.re_requirement, line)
                if m is None:
                    return False

                package = m.group(1).strip()
                version_required = (m.group(2) or "").strip()

                if version_required == "":
                    continue

                try:
                    version_installed = importlib.metadata.version(package)
                except Exception:
                    return False

                if packaging.version.parse(version_required) != packaging.version.parse(version_installed):
                    return False

        return True

    def is_installed(self, package):
        try:
            spec = find_spec(package)
        except ModuleNotFoundError:
            return False

        return spec is not None

    def download_models(self, default_model=None, previous_default_models=None):
        def ini_args():
            from adapter.args_manager import args
            return args

        args = ini_args()
        from modules.model_loader import load_file_from_url
        from modules.config import (paths_checkpoints as modelfile_path,
                                    paths_loras as lorafile_path,
                                    path_vae_approx as vae_approx_path,
                                    path_fooocus_expansion as fooocus_expansion_path,
                                    checkpoint_downloads,
                                    path_embeddings as embeddings_path,
                                    embeddings_downloads, lora_downloads)

        vae_approx_filenames = [
            ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
            ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
            ('xl-to-v1_interposer-v4.0.safetensors',
             'https://huggingface.co/lllyasviel/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors')
        ]

        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
            model_dir=fooocus_expansion_path,
            file_name='pytorch_model.bin'
        )

        if args.disable_preset_download:
            print('Skipped model download.')
            return default_model, checkpoint_downloads

        if not args.always_download_new_model:
            if not os.path.exists(os.path.join(modelfile_path[0], default_model)):
                for alternative_model_name in previous_default_models:
                    if os.path.exists(os.path.join(modelfile_path[0], alternative_model_name)):
                        print(f'You do not have [{default_model}] but you have [{alternative_model_name}].')
                        print(f'Fooocus will use [{alternative_model_name}] to avoid downloading new models, '
                              f'but you are not using the latest models.')
                        print('Use --always-download-new-model to avoid fallback and always get new models.')
                        checkpoint_downloads = {}
                        default_model = alternative_model_name
                        break

        for file_name, url in checkpoint_downloads.items():
            load_file_from_url(url=url, model_dir=modelfile_path[0], file_name=file_name)
        for file_name, url in embeddings_downloads.items():
            load_file_from_url(url=url, model_dir=embeddings_path, file_name=file_name)
        for file_name, url in lora_downloads.items():
            load_file_from_url(url=url, model_dir=lorafile_path[0], file_name=file_name)
        for file_name, url in vae_approx_filenames:
            load_file_from_url(url=url, model_dir=vae_approx_path, file_name=file_name)

        return default_model, checkpoint_downloads

    def install_dependents(self, args):
        if not args.skip_pip:
            torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu121")

            # Check if you need pip install
            if not self.requirements_met(self.requirements_file):
                self.run_pip(f"install -r \"{self.requirements_file}\"", "requirements")

            if not self.is_installed("torch") or not self.is_installed("torchvision"):
                printF(name=MasterName.get_master_name(), info="torch_index_url: " + torch_index_url).printf()
                self.run_pip(f"install torch==2.1.0 torchvision==0.16.0 --extra-index-url {torch_index_url}", "torch")

            if args.persistent and not self.is_installed("sqlalchemy"):
                self.run_pip(f"install sqlalchemy==2.0.25", "sqlalchemy")

        # Add dependent repositories to import path
        sys.path.append(self.script_path)
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    def prepare_environments(self, args) -> bool:
        if hasattr(args, "gpu_device_id") and args.gpu_device_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
            printF(name=MasterName.get_master_name(), info="Set device to: " + args.gpu_device_id).printf()

        if args.base_url is None or len(args.base_url.strip()) == 0:
            host = args.host
            if host == '0.0.0.0':
                host = '127.0.0.1'
            args.base_url = f"http://{host}:{args.port}"

        sys.argv = [sys.argv[0]]

        import modules.config as config
        import adapter.parameters as parameters
        parameters.default_inpaint_engine_version = config.default_inpaint_engine_version
        parameters.default_styles = config.default_styles
        parameters.default_base_model_name = config.default_base_model_name
        parameters.default_refiner_model_name = config.default_refiner_model_name
        parameters.default_refiner_switch = config.default_refiner_switch
        parameters.default_loras = config.default_loras
        parameters.default_cfg_scale = config.default_cfg_scale
        parameters.default_prompt_negative = config.default_prompt_negative
        parameters.default_aspect_ratio = parameters.get_aspect_ratio_value(config.default_aspect_ratio)
        parameters.available_aspect_ratios = [parameters.get_aspect_ratio_value(a) for a in
                                              config.available_aspect_ratios]

        printF(name=MasterName.get_master_name(), info="download_models").printf()
        self.download_models(default_model=default_model, previous_default_models=previous_default_models)

        self.init_task_queue(args=args)

        return True

    def init_task_queue(self, args):

        # Init task queue
        import procedure.worker as worker
        from adapter.task_queue import TaskQueue
        worker.worker_queue = TaskQueue(queue_size=args.queue_size, hisotry_size=args.queue_history,
                                        webhook_url=args.webhook_url, persistent=args.persistent)

        printF(name=MasterName.get_master_name(),
               info="[MeaVon-API] Task queue size: " + str(args.queue_size) +
                    ", queue history size: " + str(args.queue_history) + ", webhook url: " + str(
                   args.webhook_url)).printf()
        return True

    def pre_setup(self, skip_sync_repo: bool = False, disable_image_log: bool = False, skip_pip=False,
                  load_all_models: bool = False, preload_pipeline: bool = False, always_gpu: bool = False,
                  all_in_fp16: bool = False, preset: str | None = None):
        class Args(object):
            host = '127.0.0.1'
            port = 8888
            base_url = None
            sync_repo = None
            disable_image_log = False
            skip_pip = False
            preload_pipeline = False
            queue_size = 100
            queue_history = 0
            preset = None
            webhook_url = None
            persistent = False
            always_gpu = False
            all_in_fp16 = False
            gpu_device_id = None
            apikey = None

        printF(name=MasterName.get_master_name(), info="Prepare environments").printf()

        args = Args()
        if skip_sync_repo:
            args.sync_repo = 'skip'
        args.disable_image_log = disable_image_log
        args.skip_pip = skip_pip
        args.preload_pipeline = preload_pipeline
        args.always_gpu = always_gpu
        args.all_in_fp16 = all_in_fp16
        args.preset = preset

        sys.argv = [sys.argv[0]]
        if args.preset is not None:
            sys.argv.append('--preset')
            sys.argv.append(args.preset)

        if args.disable_image_log:
            sys.argv.append('--disable-image-log')

        printF(name=MasterName.get_master_name(), info="install_dependents").printf()
        self.install_dependents(args)

        import adapter.args as _
        printF(name=MasterName.get_master_name(), info="prepare_environments").printf()
        self.prepare_environments(args)

        if load_all_models:
            import modules.config as config
            from adapter.parameters import default_inpaint_engine_version
            config.downloading_upscale_model()
            config.downloading_inpaint_models(default_inpaint_engine_version)
            config.downloading_controlnet_canny()
            config.downloading_controlnet_cpds()
            config.downloading_ip_adapters()
        printF(name=MasterName.get_master_name(), info="Pre Setup Finished").printf()


if __name__ == "__main__":
    pass
