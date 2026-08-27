"""
Custom Image component for Gradio 6.14.0 - ULTRA ROBUST
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import PIL
import PIL.ImageOps
import gradio as gr
from gradio.events import EventListenerMethod
from PIL import Image as _Image

# 设置日志，方便调试
logger = logging.getLogger(__name__)

from gradio.events import Dependency

class Image(gr.Image):
    """
    Robust version: all preprocessing failures fall back to safe values.
    """

    def __init__(
        self,
        value: str | _Image.Image | np.ndarray | None = None,
        *,
        shape: tuple[int, int] | None = None,
        height: int | None = None,
        width: int | None = None,
        image_mode: Literal[
            "1", "L", "P", "RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV", "I", "F"
        ] = "RGB",
        invert_colors: bool = False,
        source: Literal["upload", "webcam", "canvas"] = "upload",
        tool: Literal["editor", "select", "sketch", "color-sketch"] | None = None,
        type: Literal["numpy", "pil", "filepath"] = "numpy",
        label: str | None = None,
        show_label: bool | None = None,
        show_download_button: bool = True,
        container: bool = True,
        scale: int | None = None,
        min_width: int = 160,
        interactive: bool | None = None,
        visible: bool = True,
        streaming: bool = False,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        mirror_webcam: bool = True,
        brush_radius: float | None = None,
        brush_color: str = "#000000",
        mask_opacity: float = 0.7,
        show_share_button: bool | None = None,
        **kwargs,
    ):
        # Store custom attributes
        self.invert_colors = invert_colors
        self.shape = shape
        self.image_mode = image_mode
        self._interpretation_segments = 16
        self.type = type
        self.source = source
        self.mirror_webcam = mirror_webcam

        # Build parent kwargs
        parent_kwargs = {
            "value": value,
            "height": height,
            "width": width,
            "sources": [source] if source else None,
            "type": type,
            "label": label,
            "show_label": show_label,
            "container": container,
            "scale": scale,
            "min_width": min_width,
            "interactive": interactive,
            "visible": visible,
            "streaming": streaming,
            "elem_id": elem_id,
            "elem_classes": elem_classes,
            **kwargs,
        }
        unsupported = [
            "tool", "brush_radius", "brush_color", "mask_opacity",
            "show_download_button", "show_share_button", "mirror_webcam"
        ]
        for key in unsupported:
            parent_kwargs.pop(key, None)

        super().__init__(**parent_kwargs)
        self.select: EventListenerMethod = self.select

    # --------------------- Preprocess (with complete error recovery) ---------------------
    def preprocess(self, x: Any) -> Any:
        """
        Convert input to expected format. Never raises exception.
        Returns None or a valid image (numpy/pil/str) as per self.type.
        """
        if x is None:
            return None

        # ---- 1. Handle dictionary inputs (sketch / ImageEditor) ----
        if isinstance(x, dict):
            # Extract the first usable image key
            for key in ("image", "background", "composite"):
                if key in x and x[key] is not None:
                    x = x[key]
                    break
            else:
                # No usable image found
                logger.warning("Dictionary input without valid image key: %s", x.keys())
                return None

        # ---- 2. Convert to PIL Image (robust) ----
        try:
            pil_img = self._to_pil(x)
            if pil_img is None:
                logger.warning("Could not convert input to PIL: %s", type(x))
                return None
        except Exception as e:
            logger.exception("Error converting to PIL: %s", e)
            return None

        # ---- 3. Apply transforms ----
        try:
            # Color mode
            pil_img = pil_img.convert(self.image_mode)

            # Crop/resize
            if self.shape is not None:
                pil_img = gr.processing_utils.resize_and_crop(pil_img, self.shape)

            # Invert
            if self.invert_colors:
                pil_img = PIL.ImageOps.invert(pil_img)

            # Mirror
            if self.source == "webcam" and self.mirror_webcam:
                pil_img = PIL.ImageOps.mirror(pil_img)

        except Exception as e:
            logger.exception("Error applying image transforms: %s", e)
            return None

        # ---- 4. Return according to self.type ----
        try:
            if self.type == "pil":
                return pil_img
            elif self.type == "numpy":
                return np.array(pil_img)
            elif self.type == "filepath":
                temp_path = gr.processing_utils.pil_to_temp_file(pil_img, format="png")
                if hasattr(self, "temp_files"):
                    self.temp_files.add(temp_path)
                return temp_path
            else:
                return pil_img  # fallback
        except Exception as e:
            logger.exception("Error finalizing image output: %s", e)
            return None

    def _to_pil(self, x: Any) -> _Image.Image | None:
        """Safe conversion to PIL Image."""
        if x is None:
            return None
        if isinstance(x, _Image.Image):
            return x
        if isinstance(x, np.ndarray):
            # Handle grayscale or RGBA correctly
            if x.ndim == 2:
                return _Image.fromarray(x, mode="L")
            elif x.ndim == 3:
                if x.shape[2] == 1:
                    return _Image.fromarray(x[:, :, 0], mode="L")
                elif x.shape[2] == 3:
                    return _Image.fromarray(x, mode="RGB")
                elif x.shape[2] == 4:
                    return _Image.fromarray(x, mode="RGBA")
            return _Image.fromarray(x)
        if isinstance(x, str):
            try:
                if Path(x).exists():
                    return _Image.open(x)
                else:
                    # assume base64
                    img_data = gr.processing_utils.decode_base64_to_image(x)
                    return _Image.open(img_data)
            except Exception:
                return None
        return None

    # --------------------- Postprocess (delegate) ---------------------
    def postprocess(self, y: Any) -> Any:
        if y is None:
            return None
        try:
            return super().postprocess(y)
        except Exception as e:
            logger.exception("Postprocess error: %s", e)
            return None

    # --------------------- Interpretation methods (unchanged) ---------------------
    def set_interpret_parameters(self, segments: int = 16):
        self._interpretation_segments = segments
        return self

    def tokenize(self, x: str):
        try:
            segments_slic, resized_and_cropped_image = self._segment_by_slic(x)
            tokens, masks, leave_one_out_tokens = [], [], []
            replace_color = np.mean(resized_and_cropped_image, axis=(0, 1))
            for segment_value in np.unique(segments_slic):
                mask = segments_slic == segment_value
                image_screen = np.copy(resized_and_cropped_image)
                image_screen[segments_slic == segment_value] = replace_color
                leave_one_out_tokens.append(
                    gr.processing_utils.encode_array_to_base64(image_screen)
                )
                token = np.copy(resized_and_cropped_image)
                token[segments_slic != segment_value] = 0
                tokens.append(token)
                masks.append(mask)
            return tokens, leave_one_out_tokens, masks
        except Exception as e:
            logger.exception("tokenize error: %s", e)
            return [], [], []

    def get_masked_inputs(self, tokens, binary_mask_matrix):
        try:
            masked_inputs = []
            for binary_mask_vector in binary_mask_matrix:
                masked_input = np.zeros_like(tokens[0], dtype=int)
                for token, b in zip(tokens, binary_mask_vector):
                    masked_input = masked_input + token * int(b)
                masked_inputs.append(gr.processing_utils.encode_array_to_base64(masked_input))
            return masked_inputs
        except Exception as e:
            logger.exception("get_masked_inputs error: %s", e)
            return []

    def get_interpretation_scores(self, x, neighbors, scores, masks, tokens=None, **kwargs):
        try:
            x = gr.processing_utils.decode_base64_to_image(x)
            if self.shape is not None:
                x = gr.processing_utils.resize_and_crop(x, self.shape)
            x = np.array(x)
            output_scores = np.zeros((x.shape[0], x.shape[1]))
            for score, mask in zip(scores, masks):
                output_scores += score * mask
            max_val, min_val = np.max(output_scores), np.min(output_scores)
            if max_val > 0:
                output_scores = (output_scores - min_val) / (max_val - min_val)
            return output_scores.tolist()
        except Exception as e:
            logger.exception("get_interpretation_scores error: %s", e)
            return [[]]

    def _segment_by_slic(self, x: str):
        try:
            from skimage.segmentation import slic
        except ImportError:
            raise ValueError("scikit-image required")
        x = gr.processing_utils.decode_base64_to_image(x)
        if self.shape is not None:
            x = gr.processing_utils.resize_and_crop(x, self.shape)
        arr = np.array(x)
        try:
            segments = slic(arr, self._interpretation_segments, compactness=10, sigma=1, start_label=1)
        except TypeError:
            segments = slic(arr, self._interpretation_segments, compactness=10, sigma=1)
        return segments, arr

    @staticmethod
    def update(**kwargs):
        return gr.update(**kwargs)

    def style(self, **kwargs):
        warnings.warn("'style()' is deprecated.", DeprecationWarning, stacklevel=2)
        return self
    from typing import Callable, Literal, Sequence, Any, TYPE_CHECKING
    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer
        from gradio.components.base import Component