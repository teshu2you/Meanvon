import os
import re
import json
import typing

import numpy as np
import datetime
import random
import math
import os
import cv2
import modules
from PIL import Image
from datetime import datetime, timedelta
import torch
from omegaconf import OmegaConf
import sys
from hashlib import sha256
import modules.sdxl_styles
from util.printf import printF, MasterName

sys.path.append("repositories/generative-models")

from sgm.util import instantiate_from_config
import ldm_patched.modules.model_management as model_management

from modules import shared
from modules.paths_internal import cwd, script_path


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


def truncate_path(target_path, base_path=cwd):
    abs_target, abs_base = os.path.abspath(target_path), os.path.abspath(base_path)
    try:
        if os.path.commonpath([abs_target, abs_base]) == abs_base:
            return os.path.relpath(abs_target, abs_base)
    except ValueError:
        pass
    return abs_target


class MassFileListerCachedDir:
    """A class that caches file metadata for a specific directory."""

    def __init__(self, dirname):
        self.files = None
        self.files_cased = None
        self.dirname = dirname

        stats = ((x.name, x.stat(follow_symlinks=False)) for x in os.scandir(self.dirname))
        files = [(n, s.st_mtime, s.st_ctime) for n, s in stats]
        self.files = {x[0].lower(): x for x in files}
        self.files_cased = {x[0]: x for x in files}

    def update_entry(self, filename):
        """Add a file to the cache"""
        file_path = os.path.join(self.dirname, filename)
        try:
            stat = os.stat(file_path)
            entry = (filename, stat.st_mtime, stat.st_ctime)
            self.files[filename.lower()] = entry
            self.files_cased[filename] = entry
        except FileNotFoundError as e:
            print(f'MassFileListerCachedDir.add_entry: "{file_path}" {e}')


class MassFileLister:
    """A class that provides a way to check for the existence and mtime/ctile of files without doing more than one stat call per file."""

    def __init__(self):
        self.cached_dirs = {}

    def find(self, path):
        """
        Find the metadata for a file at the given path.

        Returns:
            tuple or None: A tuple of (name, mtime, ctime) if the file exists, or None if it does not.
        """

        dirname, filename = os.path.split(path)

        cached_dir = self.cached_dirs.get(dirname)
        if cached_dir is None:
            cached_dir = MassFileListerCachedDir(dirname)
            self.cached_dirs[dirname] = cached_dir

        stats = cached_dir.files_cased.get(filename)
        if stats is not None:
            return stats

        stats = cached_dir.files.get(filename.lower())
        if stats is None:
            return None

        try:
            os_stats = os.stat(path, follow_symlinks=False)
            return filename, os_stats.st_mtime, os_stats.st_ctime
        except Exception:
            return None

    def exists(self, path):
        """Check if a file exists at the given path."""

        return self.find(path) is not None

    def mctime(self, path):
        """
        Get the modification and creation times for a file at the given path.

        Returns:
            tuple: A tuple of (mtime, ctime) if the file exists, or (0, 0) if it does not.
        """

        stats = self.find(path)
        return (0, 0) if stats is None else stats[1:3]

    def reset(self):
        """Clear the cache of all directories."""
        self.cached_dirs.clear()

    def update_file_entry(self, path):
        """Update the cache for a specific directory."""
        dirname, filename = os.path.split(path)
        if cached_dir := self.cached_dirs.get(dirname):
            cached_dir.update_entry(filename)


def topological_sort(dependencies):
    """Accepts a dictionary mapping name to its dependencies, returns a list of names ordered according to dependencies.
    Ignores errors relating to missing dependencies or circular dependencies
    """

    visited = {}
    result = []

    def inner(name):
        visited[name] = True

        for dep in dependencies.get(name, []):
            if dep in dependencies and dep not in visited:
                inner(dep)

        result.append(name)

    for depname in dependencies:
        if depname not in visited:
            inner(depname)

    return result


def open_folder(path):
    """Open a folder in the file manager of the respect OS."""
    # import at function level to avoid potential issues
    import platform
    import subprocess
    import sys

    import gradio as gr

    if not os.path.exists(path):
        msg = f'Folder "{path}" does not exist. after you save an image, the folder will be created.'
        print(msg)
        gr.Info(msg)
        return
    elif not os.path.isdir(path):
        msg = f"""
WARNING
An open_folder request was made with an path that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {path}
"""
        print(msg, file=sys.stderr)
        gr.Warning(msg)
        return

    path = os.path.normpath(path)
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    elif "microsoft-standard-WSL2" in platform.uname().release:
        subprocess.Popen(["explorer.exe", subprocess.check_output(["wslpath", "-w", path])])
    else:
        subprocess.Popen(["xdg-open", path])


def load_file_from_url(
    url: str,
    *,
    model_dir: str,
    progress: bool = True,
    file_name: str | None = None,
    hash_prefix: str | None = None,
    re_download: bool = False,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.
    Returns the path to the downloaded file.

    file_name: if specified, it will be used as the filename, otherwise the filename will be extracted from the url.
        file is downloaded to {file_name}.tmp then moved to the final location after download is complete.
    hash_prefix: sha256 hex string, if provided, the hash of the downloaded file will be checked against this prefix.
        if the hash does not match, the temporary file is deleted and a ValueError is raised.
    re_download: forcibly re-download the file even if it already exists.
    """
    from urllib.parse import urlparse

    import requests
    from tqdm import tqdm

    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)

    cached_file = os.path.abspath(os.path.join(model_dir, file_name))

    if re_download or not os.path.exists(cached_file):
        os.makedirs(model_dir, exist_ok=True)
        temp_file = os.path.join(model_dir, f"{file_name}.tmp")
        print(f'\nDownloading: "{url}" to {cached_file}')
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=file_name, disable=not progress) as progress_bar:
            with open(temp_file, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))

        if hash_prefix and not compare_sha256(temp_file, hash_prefix):
            print(f"Hash mismatch for {temp_file}. Deleting the temporary file.")
            os.remove(temp_file)
            raise ValueError(f"File hash does not match the expected hash prefix {hash_prefix}!")

        os.rename(temp_file, cached_file)
    return cached_file


def compare_sha256(file_path: str, hash_prefix: str) -> bool:
    """Check if the SHA256 hash of the file matches the given prefix."""
    import hashlib

    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest().startswith(hash_prefix.strip().lower())



config_dict = {}
model_preset_config_dict = {}


def get_model_preset_config_item_or_set_default(key, default_value, validator, disable_empty_as_none=False):
    global model_preset_config_dict
    if key not in model_preset_config_dict:
        model_preset_config_dict[key] = default_value
        return default_value

    v = model_preset_config_dict.get(key, None)
    if not disable_empty_as_none:
        if v is None or v == '':
            v = 'None'
    if validator(v):
        return v
    else:
        model_preset_config_dict[key] = default_value
        return default_value


def get_config_item_or_set_default(key, default_value, validator, disable_empty_as_none=False):
    global config_dict
    if key not in config_dict:
        config_dict[key] = default_value
        return default_value

    v = config_dict.get(key, None)
    if not disable_empty_as_none:
        if v is None or v == '':
            v = 'None'
    if validator(v):
        return v
    else:
        config_dict[key] = default_value
        return default_value


def get_config_or_set_default(key, default):
    global config_dict
    v = config_dict.get(key, None)
    if not isinstance(v, str):
        v = default
    dp = v if os.path.isabs(v) else os.path.abspath(os.path.join(os.path.dirname(__file__), v))
    if not os.path.exists(dp) or not os.path.isdir(dp):
        os.makedirs(dp, exist_ok=True)
    config_dict[key] = dp
    return dp


def image_is_generated_in_current_ui(image, ui_width, ui_height):
    H, W, C = image.shape

    if H < ui_height:
        return False

    if W < ui_width:
        return False

    # k1 = float(H) / float(W)
    # k2 = float(ui_height) / float(ui_width)
    # d = abs(k1 - k2)
    #
    # if d > 0.01:
    #     return False

    return True


LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
wildcards_path = get_config_or_set_default('wildcards_path', '../wildcards/')
HASH_SHA256_LENGTH = 10


def erode_or_dilate(x, k):
    k = int(k)
    if k > 0:
        return cv2.dilate(x, kernel=np.ones(shape=(3, 3), dtype=np.uint8), iterations=k)
    if k < 0:
        return cv2.erode(x, kernel=np.ones(shape=(3, 3), dtype=np.uint8), iterations=-k)
    return x


def resample_image(im, width, height):
    im = Image.fromarray(im)
    im = im.resize((int(width), int(height)), resample=LANCZOS)
    return np.array(im)


def resize_image(im, width, height, resize_mode=1):
    """
    Resizes an image with the specified resize_mode, width, and height.

    Args:
        resize_mode: The mode to use when resizing the image.
            0: Resize the image to the specified width and height.
            1: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
            2: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
        im: The image to resize.
        width: The width to resize the image to.
        height: The height to resize the image to.
    """

    im = Image.fromarray(im)

    def resize(im, w, h):
        return im.resize((w, h), resample=LANCZOS)

    if resize_mode == 0:
        res = resize(im, width, height)

    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            if fill_height > 0:
                res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)),
                          box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            if fill_width > 0:
                res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)),
                          box=(fill_width + src_w, 0))

    return np.array(res)


def get_shape_ceil(h, w):
    return math.ceil(((h * w) ** 0.5) / 64.0) * 64.0


def get_image_shape_ceil(im):
    H, W = im.shape[:2]
    return get_shape_ceil(H, W)


def set_image_shape_ceil(im, shape_ceil):
    shape_ceil = float(shape_ceil)

    H_origin, W_origin, _ = im.shape
    H, W = H_origin, W_origin

    for _ in range(256):
        current_shape_ceil = get_shape_ceil(H, W)
        if abs(current_shape_ceil - shape_ceil) < 0.1:
            break
        k = shape_ceil / current_shape_ceil
        H = int(round(float(H) * k / 64.0) * 64)
        W = int(round(float(W) * k / 64.0) * 64)

    if H == H_origin and W == W_origin:
        return im

    return resample_image(im, width=W, height=H)


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def remove_empty_str(items, default=None):
    items = [x for x in items if x != ""]
    if len(items) == 0 and default is not None:
        return [default]
    return items


def join_prompts(*args, **kwargs):
    prompts = [str(x) for x in args if str(x) != ""]
    if len(prompts) == 0:
        return ""
    if len(prompts) == 1:
        return prompts[0]
    return ', '.join(prompts)


def generate_temp_filename(folder='./outputs/', extension='png', base=None, use_new=True):
    current_time = datetime.now()
    date_string = current_time.strftime("%Y-%m-%d")
    prefix_time_string = current_time.strftime("%Y%m%d%H%M%S")
    printF(name=MasterName.get_master_name(), info="[Info] Prefix Time String:{}".format(prefix_time_string)).printf()
    if base is None:
        time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        random_number = random.randint(1000, 9999)
        filename = f"{time_string}_{random_number}.{extension}"
    else:
        if use_new:
            filename = f"{base}_{prefix_time_string}.{extension}"
        else:
            filename = f"{os.path.splitext(base)[0]}.{extension}"
    result = os.path.join(folder, date_string, filename)
    return date_string, os.path.abspath(os.path.realpath(result)), filename


def get_files_from_folder(folder_path, exensions=None, name_filter=None):
    if not os.path.isdir(folder_path):
        raise ValueError("Folder path is not a valid directory.")

    filenames = []

    for root, dirs, files in os.walk(folder_path, topdown=False):
        relative_path = os.path.relpath(root, folder_path)
        if relative_path == ".":
            relative_path = ""
        for filename in sorted(files, key=lambda s: s.casefold()):
            _, file_extension = os.path.splitext(filename)
            if (exensions is None or file_extension.lower() in exensions) and (name_filter is None or name_filter.lower() in _.lower()):
                path = os.path.join(relative_path, filename)
                filenames.append(path)
    return filenames


def calculate_sha256(filename, length=HASH_SHA256_LENGTH) -> str:
    hash_sha256 = sha256()
    blksize = 1024 * 1024

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)

    res = hash_sha256.hexdigest()
    return res[:length] if length else res


def quote(text):
    if ',' not in str(text) and '\n' not in str(text) and ':' not in str(text):
        return text

    return json.dumps(text, ensure_ascii=False)


def unquote(text):
    if len(text) == 0 or text[0] != '"' or text[-1] != '"':
        return text

    try:
        return json.loads(text)
    except Exception:
        return text


def unwrap_style_text_from_prompt(style_text, prompt):
    """
    Checks the prompt to see if the style text is wrapped around it. If so,
    returns True plus the prompt text without the style text. Otherwise, returns
    False with the original prompt.

    Note that the "cleaned" version of the style text is only used for matching
    purposes here. It isn't returned; the original style text is not modified.
    """
    stripped_prompt = prompt
    stripped_style_text = style_text
    if "{prompt}" in stripped_style_text:
        # Work out whether the prompt is wrapped in the style text. If so, we
        # return True and the "inner" prompt text that isn't part of the style.
        try:
            left, right = stripped_style_text.split("{prompt}", 2)
        except ValueError as e:
            # If the style text has multple "{prompt}"s, we can't split it into
            # two parts. This is an error, but we can't do anything about it.
            print(f"Unable to compare style text to prompt:\n{style_text}")
            print(f"Error: {e}")
            return False, prompt, ''

        left_pos = stripped_prompt.find(left)
        right_pos = stripped_prompt.find(right)
        if 0 <= left_pos < right_pos:
            real_prompt = stripped_prompt[left_pos + len(left):right_pos]
            prompt = stripped_prompt.replace(left + real_prompt + right, '', 1)
            if prompt.startswith(", "):
                prompt = prompt[2:]
            if prompt.endswith(", "):
                prompt = prompt[:-2]
            return True, prompt, real_prompt
    else:
        # Work out whether the given prompt starts with the style text. If so, we
        # return True and the prompt text up to where the style text starts.
        if stripped_prompt.endswith(stripped_style_text):
            prompt = stripped_prompt[: len(stripped_prompt) - len(stripped_style_text)]
            if prompt.endswith(", "):
                prompt = prompt[:-2]
            return True, prompt, prompt

    return False, prompt, ''


def extract_styles_from_prompt(prompt, negative_prompt):
    extracted = []
    applicable_styles = []

    for style_name, (style_prompt, style_negative_prompt) in modules.sdxl_styles.styles.items():
        applicable_styles.append(
            PromptStyle(name=style_name, prompt=style_prompt, negative_prompt=style_negative_prompt))

    real_prompt = ''

    while True:
        found_style = None

        for style in applicable_styles:
            is_match, new_prompt, new_neg_prompt, new_real_prompt = extract_original_prompts(
                style, prompt, negative_prompt
            )
            if is_match:
                found_style = style
                prompt = new_prompt
                negative_prompt = new_neg_prompt
                if real_prompt == '' and new_real_prompt != '' and new_real_prompt != prompt:
                    real_prompt = new_real_prompt
                break

        if not found_style:
            break

        applicable_styles.remove(found_style)
        extracted.append(found_style.name)

    # add prompt expansion if not all styles could be resolved
    if prompt != '':
        if real_prompt != '':
            extracted.append(modules.sdxl_styles.fooocus_expansion)
        else:
            # find real_prompt when only prompt expansion is selected
            first_word = prompt.split(', ')[0]
            first_word_positions = [i for i in range(len(prompt)) if prompt.startswith(first_word, i)]
            if len(first_word_positions) > 1:
                real_prompt = prompt[:first_word_positions[-1]]
                extracted.append(modules.sdxl_styles.fooocus_expansion)
                if real_prompt.endswith(', '):
                    real_prompt = real_prompt[:-2]

    return list(reversed(extracted)), real_prompt, negative_prompt


class PromptStyle(typing.NamedTuple):
    name: str
    prompt: str
    negative_prompt: str


def is_json(data: str) -> bool:
    try:
        loaded_json = json.loads(data)
        assert isinstance(loaded_json, dict)
    except (ValueError, AssertionError):
        return False
    return True


def get_file_from_folder_list(name, folders):
    # print(f"name : {name}")
    # print(f"folders : {folders}")

    for folder in folders:
        filename = os.path.abspath(os.path.realpath(os.path.join(folder, name)))
        if os.path.isfile(filename):
            return filename

    if isinstance(folders, list):
        return os.path.abspath(os.path.realpath(os.path.join(folders[0], name)))
    elif isinstance(folders, str):
        return os.path.abspath(os.path.realpath(os.path.join(folders, name)))



def extract_original_prompts(style, prompt, negative_prompt):
    """
    Takes a style and compares it to the prompt and negative prompt. If the style
    matches, returns True plus the prompt and negative prompt with the style text
    removed. Otherwise, returns False with the original prompt and negative prompt.
    """
    if not style.prompt and not style.negative_prompt:
        return False, prompt, negative_prompt

    match_positive, extracted_positive, real_prompt = unwrap_style_text_from_prompt(
        style.prompt, prompt
    )
    if not match_positive:
        return False, prompt, negative_prompt, ''

    match_negative, extracted_negative, _ = unwrap_style_text_from_prompt(
        style.negative_prompt, negative_prompt
    )
    if not match_negative:
        return False, prompt, negative_prompt, ''

    return True, extracted_positive, extracted_negative, real_prompt


def get_var_name(var):
    import inspect
    _vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in _vars if var_val is var]


def get_log_path(time):
    current_dir = os.path.abspath(os.curdir)
    outputs_dir = os.path.abspath(modules.config.temp_outputs_path)
    if outputs_dir.startswith(current_dir):
        folder = os.path.relpath(outputs_dir, current_dir)
    else:
        folder = outputs_dir
    date_string = time.strftime("%Y-%m-%d")
    folder += ""
    return os.path.join(folder, date_string, "Gallery_Images_" + date_string + '.html')


def get_current_log_path():
    time = datetime.now()
    return get_log_path(time)


def show_cuda_info():
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        c_gpu_idx = torch.cuda.current_device()
        c_gpu_name = torch.cuda.get_device_name()

        total_mem = torch.cuda.get_device_properties(c_gpu_idx).total_memory / (1024 ** 3)
        used_mem = torch.cuda.memory_allocated(c_gpu_idx) / (1024 ** 3)
        free_mem = total_mem - used_mem

        print("**" * 100)
        print(f"{n_gpu} GPU devices are available")
        print(f"current gpu index      : {c_gpu_idx}")
        print(f"current gpu device name: {c_gpu_name}")
        print(f"total GPU memory       : {total_mem:.2f} GB")
        print(f"used GPU memory        : {used_mem:.2f} GB")
        print(f"free GPU memory        : {free_mem:.2f} GB")
        print("**" * 100)
    else:
        print(f"CUDA is unavailable!")


def free_cuda_cache():
    print(f"cuda cache freed!")
    model_management.soft_empty_cache()
    show_cuda_info()


def free_cuda_mem():
    print(f"all models unloaded!")
    model_management.unload_all_models()
    show_cuda_info()


def get_previous_log_path():
    time = datetime.now() - timedelta(days=1)
    return get_log_path(time)


def load_model(config: str, device: str, num_frames: int, num_steps: int):
    config = OmegaConf.load(config)
    config.model.params.conditioner_config.params.emb_models[
        0].params.open_clip_embedding_config.params.init_device = device
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    print(f'video factory config: {config}')
    with torch.device(device):
        model = instantiate_from_config(config.model).to(device).eval().requires_grad_(False)

    return model


def ordinal_suffix(number: int) -> str:
    return 'th' if 10 <= number % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')


def makedirs_with_log(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        print(f'Directory {path} could not be created, reason: {error}')


def get_enabled_loras(loras: list) -> list:
    return [[lora[1], lora[2]] for lora in loras if lora[0]]

def apply_native_tiling(model, enable=False):
    """
    Natively enables or disables seamless
    circular padding on all Conv2D layers.
    Recursively unpacks any custom wrapper
    classes down to the core PyTorch module.
    """
    if model is None:
        return

    # 1. Recursively unpack any nesting
    # (StableDiffusionModel -> ModelPatcher -> BaseModel)
    # until we locate the actual PyTorch nn.Module
    # (the diffusion_model)
    active_module = model
    while hasattr(active_module, 'unet') or hasattr(active_module, 'model') or hasattr(active_module, 'inner_model') or hasattr(active_module, 'diffusion_model'):
        if hasattr(active_module, 'diffusion_model'):
            active_module = active_module.diffusion_model
            break
        elif hasattr(active_module, 'inner_model'):
            active_module = active_module.inner_model
        elif hasattr(active_module, 'unet'):
            active_module = active_module.unet
        elif hasattr(active_module, 'model'):
            active_module = active_module.model
        else:
            break

    # 2. Ensure the resolved object actually
    # has the .modules() method
    if not hasattr(active_module, 'modules'):
        print(f"[Utility] Apply_native_tiling could not resolve PyTorch modules from {type(model)}")
        try:
            available_attrs = dir(model)
            print('Available attributes on', f'{type(model).__name__}: {available_attrs}')
        except Exception as e:
            print(f'Failed to extract any attributes:', e)
        return

    if enable:
        target_mode = 'circular'
        print('[Utility] Tiling enabled')
    else:
        target_mode = 'zeros'

    # 3. Safely apply the padding mode
    count = 0
    for module in active_module.modules():
        if isinstance(module, torch.nn.Conv2d):
            module.padding_mode = target_mode
            count += 1

    return
