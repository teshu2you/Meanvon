import os
import re
import sys
from typing import TYPE_CHECKING

import gradio as gr

from backend import memory_management
from modules import options, shared_cmd_options, shared_gradio_themes, shared_items, util
from modules.paths_internal import data_path, extensions_builtin_dir, extensions_dir, models_path, script_path  # noqa: F401

if TYPE_CHECKING:
    from backend.diffusion_engine.base import ForgeDiffusionEngine
    from modules import face_restoration, memmon, shared_state, shared_total_tqdm, styles, upscaler

cmd_opts = shared_cmd_options.cmd_opts
parser = shared_cmd_options.parser

batch_cond_uncond = True  # shared.opts.batch_cond_uncond
parallel_processing_allowed = True
styles_filename = cmd_opts.styles_file = cmd_opts.styles_file if len(cmd_opts.styles_file) > 0 else [os.path.join(data_path, "styles.csv"), os.path.join(data_path, "styles_integrated.csv")]
config_filename = cmd_opts.ui_settings_file
hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}

demo: gr.Blocks = None

device: str = None

xformers_available = memory_management.xformers_enabled()

state: "shared_state.State" = None

prompt_styles: "styles.StyleDatabase" = None

face_restorers: list["face_restoration.FaceRestoration"] = []

options_templates: dict = None
opts: options.Options = None
restricted_opts: set[str] = None

sd_model: "ForgeDiffusionEngine" = None

settings_components: dict = None
"""assigned from ui.py, a mapping on setting names to gradio components responsible for those settings"""

tab_names: list[str] = []

latent_upscale_default_mode = "Latent"
latent_upscale_modes = {
    "Latent": {"mode": "bilinear", "antialias": False},
    "Latent (antialiased)": {"mode": "bilinear", "antialias": True},
    "Latent (bicubic)": {"mode": "bicubic", "antialias": False},
    "Latent (bicubic antialiased)": {"mode": "bicubic", "antialias": True},
    "Latent (nearest)": {"mode": "nearest", "antialias": False},
    "Latent (nearest-exact)": {"mode": "nearest-exact", "antialias": False},
}

sd_upscalers: list["upscaler.Upscaler"] = []

progress_print_out = sys.stdout

gradio_theme = gr.themes.Base()

total_tqdm: "shared_total_tqdm.TotalTQDM" = None

mem_mon: "memmon.MemUsageMonitor" = None

options_section = options.options_section
OptionInfo = options.OptionInfo
OptionHTML = options.OptionHTML

def natural_sort_key(s: str, regex=re.compile(r"([0-9]+)")):
    return [[int(text) if text.isdigit() else text.lower() for text in regex.split(part)] for part in os.path.splitext(s)]

def listfiles(dirname):
    filenames = [os.path.join(dirname, x) for x in sorted(os.listdir(dirname), key=natural_sort_key) if not x.startswith(".")]
    return [file for file in filenames if os.path.isfile(file)]


def html_path(filename):
    return os.path.join(script_path, "html", filename)


def html(filename):
    path = html_path(filename)

    try:
        with open(path, encoding="utf8") as file:
            return file.read()
    except OSError:
        return ""


def walk_files(path, allowed_extensions=None):
    if not os.path.exists(path):
        return

    if allowed_extensions is not None:
        allowed_extensions = set(allowed_extensions)

    items = list(os.walk(path, followlinks=True))
    items = sorted(items, key=lambda x: natural_sort_key(x[0]))

    for root, _, files in items:
        for filename in sorted(files, key=natural_sort_key):
            if allowed_extensions is not None:
                _, ext = os.path.splitext(filename)
                if ext.lower() not in allowed_extensions:
                    continue

            if not shared.opts.list_hidden_files and ("/." in root or "\\." in root):
                continue

            yield os.path.join(root, filename)

reload_gradio_theme = shared_gradio_themes.reload_gradio_theme

list_checkpoint_tiles = shared_items.list_checkpoint_tiles
refresh_checkpoints = shared_items.refresh_checkpoints
list_samplers = shared_items.list_samplers

hf_endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co")
