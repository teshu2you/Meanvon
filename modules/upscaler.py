import os
from abc import abstractmethod

from PIL import Image

from modules import devices, modelloader, shared
from modules.images import LANCZOS, NEAREST
from modules.shared import models_path, opts

# hardcode
UPSCALE_ITERATIONS = 4

def perform_upscale(*args, **kwargs):
    """
    Compatibility shim for Fooocus-style inpaint_worker.

    Supported forms:
      perform_upscale(img)                                # default x2
      perform_upscale(img, scale)                         # numeric scale
      perform_upscale(img, upscaler_name, scale)          # named upscaler
      perform_upscale(img, scale=..., upscaler_name=...)  # kwargs

    Handles: PIL.Image / torch.Tensor / numpy.ndarray
    """
    from modules import shared as _shared

    if not args:
        raise TypeError("perform_upscale: requires at least one positional argument (image)")

    img = args[0]
    scale = kwargs.get("scale", 2.0)
    upscaler_name = kwargs.get("upscaler_name", None)

    if len(args) >= 2:
        a1 = args[1]
        if isinstance(a1, (int, float)):
            scale = float(a1)
        elif isinstance(a1, str):
            upscaler_name = a1
            if len(args) >= 3:
                scale = float(args[2])

    scale = float(scale)

    # --- torch latent upscale ---
    try:
        import torch
        if isinstance(img, torch.Tensor):
            import torch.nn.functional as _F
            if img.dim() == 4:
                return _F.interpolate(img, scale_factor=scale, mode="bilinear", align_corners=False)
            if img.dim() == 3:
                return _F.interpolate(img.unsqueeze(0), scale_factor=scale,
                                      mode="bilinear", align_corners=False).squeeze(0)
            return img
    except ImportError:
        pass

    # --- PIL image upscale (named upscaler when available) ---
    if isinstance(img, Image.Image):
        upscalers = getattr(_shared, "sd_upscalers", None) or []
        if upscaler_name:
            for u in upscalers:
                if getattr(u, "name", None) == upscaler_name and getattr(u, "data_path", None):
                    return u.scaler.upscale(img, scale, u.data_path)
        return img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            LANCZOS,
        )

    # --- numpy array ---
    try:
        import numpy as np
        if isinstance(img, np.ndarray):
            w = max(1, round(img.shape[1] * scale))
            h = max(1, round(img.shape[0] * scale))
            return np.array(Image.fromarray(img).resize((w, h), LANCZOS))
    except ImportError:
        pass

    raise TypeError(f"perform_upscale: unsupported image type {type(img)!r}")

class Upscaler:
    name = None
    model_path = None
    model_name = None
    model_url = None
    enable = True
    filter = None
    model = None
    user_path = None
    scalers: list["UpscalerData"] = []
    tile = True

    def __init__(self, create_dirs=False):
        self.scale: int = 1
        self.tile_size: int = opts.ESRGAN_tile
        self.tile_pad: int = opts.ESRGAN_tile_overlap
        self.device = devices.device_esrgan
        self.model_download_path: str = None
        self.img = None
        self.output = None

        if self.model_path is None and self.name:
            self.model_path = os.path.join(models_path, self.name)
        if self.model_path:
            if create_dirs and not os.path.isdir(self.model_path):
                os.makedirs(self.model_path)

    @abstractmethod
    def do_upscale(self, img: Image.Image, selected_model: str):
        raise NotImplementedError

    @abstractmethod
    def load_model(self, path: str):
        raise NotImplementedError

    def upscale(self, img: Image.Image, scale: float, selected_model: str = None):
        dest_w: int = round(img.width * scale / 8) * 8
        dest_h: int = round(img.height * scale / 8) * 8

        for _ in range(UPSCALE_ITERATIONS):
            if shared.state.interrupted:
                break

            _orig = img.size

            img = self.do_upscale(img, selected_model)

            if (img.width >= dest_w and img.height >= dest_h) or img.size == _orig:
                break

        if (img.width != dest_w) or (img.height != dest_h):
            img = img.resize((dest_w, dest_h), LANCZOS)

        return img

    def find_models(self, ext_filter=None) -> list[str]:
        return modelloader.load_models(
            model_path=self.model_path,
            model_url=self.model_url,
            command_path=self.user_path,
            ext_filter=ext_filter,
        )


class UpscalerData:
    name: str
    data_path: str
    scaler: Upscaler
    scale: int
    model: None

    def __init__(
        self,
        name: str,
        path: str,
        upscaler: Upscaler = None,
        scale: int = 4,
        model=None,
    ):
        self.name = name
        self.data_path = path
        self.scaler = upscaler
        self.scale = scale
        self.model = model

    def __repr__(self):
        return f"<UpscalerData name={self.name} path={self.data_path} scale={self.scale}>"


class UpscalerNone(Upscaler):
    def __init__(self, dirname=None):
        super().__init__(False)
        self.name = "None"
        self.scalers = [UpscalerData("None", None, self)]

    def load_model(self, _):
        return

    def do_upscale(self, img: Image.Image, *args, **kwargs):
        return img


class UpscalerLanczos(Upscaler):
    def __init__(self, dirname=None):
        super().__init__(False)
        self.name = "Lanczos"
        self.scalers = [UpscalerData("Lanczos", None, self)]

    def load_model(self, _):
        return

    def do_upscale(self, img: Image.Image, *args, **kwargs):
        return img.resize(
            size=(int(img.width * self.scale), int(img.height * self.scale)),
            resample=LANCZOS,
        )


class UpscalerNearest(Upscaler):
    def __init__(self, dirname=None):
        super().__init__(False)
        self.name = "Nearest"
        self.scalers = [UpscalerData("Nearest", None, self)]

    def load_model(self, _):
        return

    def do_upscale(self, img: Image.Image, *args, **kwargs):
        return img.resize(
            size=(int(img.width * self.scale), int(img.height * self.scale)),
            resample=NEAREST,
        )
