from enum import Enum
from typing import Dict, List, Tuple
import numpy as np
from modules.flags import Performance
from modules import config

default_inpaint_engine_version = 'v2.6'

default_styles = config.default_styles
default_base_model_name = config.default_base_model_name
default_refiner_model_name = config.default_refiner_model_name
default_refiner_switch = config.default_refiner_switch
default_loras = config.default_loras
default_cfg_scale = config.default_cfg_scale
default_prompt_negative = config.default_prompt_negative
default_aspect_ratio = config.default_aspect_ratio
default_sampler = config.default_sampler
default_scheduler = config.default_scheduler

available_aspect_ratios = [
    '704*1408',
    '704*1344',
    '768*1344',
    '768*1280',
    '832*1216',
    '832*1152',
    '896*1152',
    '896*1088',
    '960*1088',
    '960*1024',
    '1024*1024',
    '1024*960',
    '1088*960',
    '1088*896',
    '1152*896',
    '1152*832',
    '1216*832',
    '1280*768',
    '1344*768',
    '1344*704',
    '1408*704',
    '1472*704',
    '1536*640',
    '1600*640',
    '1664*576',
    '1728*576',
]

uov_methods = [
    'Disabled', 'Vary (Subtle)', 'Vary (Strong)', 'Upscale (1.5x)', 'Upscale (2x)', 'Upscale (Fast 2x)',
    'Upscale (Custom)'
]

outpaint_expansions = [
    'Left', 'Right', 'Top', 'Bottom'
]


def get_aspect_ratio_value(label: str) -> str:
    return label.split(' ')[0].replace('×', '*')


class GenerationFinishReason(str, Enum):
    success = 'SUCCESS'
    queue_is_full = 'QUEUE_IS_FULL'
    user_cancel = 'USER_CANCEL'
    error = 'ERROR'


class ImageGenerationResult(object):
    def __init__(self, im: str | None, seed: str, finish_reason: GenerationFinishReason):
        self.im = im
        self.seed = seed
        self.finish_reason = finish_reason


class ImageGenerationParams(object):
    # meanvon
    def __init__(self, prompt: str = "",
                 negative_prompt: str = "",
                 style_selections: List[str] = default_styles,
                 performance_selection: str = Performance.SPEED,
                 aspect_ratios_selection: str = "512×512 <span style=\"color: grey;\"> ∣ 1:1</span>",
                 image_number: int = 1,
                 image_seed: int | None = 1,
                 sharpness: float = 2.0,
                 guidance_scale: float = 4.0,
                 base_model_name: str = default_base_model_name,
                 refiner_model_name: str = "None",
                 refiner_switch: float = 0.8,
                 loras: List[any] = [[True, 'None', 1.0], [True, 'None', 1.0], [True, 'None', 1.0],
                                                        [True, 'None', 1.0], [True, 'None', 1.0]],
                 uov_input_image: np.ndarray | None = None,
                 uov_method: str = "Disabled",
                 upscale_value: float | None = None,
                 outpaint_selections: List[str] = ["","","",""],
                 outpaint_distance_left: int = 1,
                 outpaint_distance_right: int = 1,
                 outpaint_distance_top: int = 1,
                 outpaint_distance_bottom: int = 1,
                 inpaint_input_image: Dict[str, np.ndarray] | None = None,
                 inpaint_additional_prompt: str | None = None,
                 image_prompts: List[Tuple[np.ndarray, float, float, str]] = [],
                 advanced_params: List[any] | None = None,
                 save_extension: str = "png",
                 require_base64: bool = False):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.style_selections = style_selections
        self.performance_selection = performance_selection
        self.aspect_ratios_selection = aspect_ratios_selection
        self.image_number = image_number
        self.image_seed = image_seed
        self.sharpness = sharpness
        self.guidance_scale = guidance_scale
        self.base_model_name = base_model_name
        self.refiner_model_name = refiner_model_name
        self.refiner_switch = refiner_switch
        self.loras = loras
        self.uov_input_image = uov_input_image
        self.uov_method = uov_method
        self.upscale_value = upscale_value
        self.outpaint_selections = outpaint_selections
        self.outpaint_distance_left = outpaint_distance_left
        self.outpaint_distance_right = outpaint_distance_right
        self.outpaint_distance_top = outpaint_distance_top
        self.outpaint_distance_bottom = outpaint_distance_bottom
        self.inpaint_input_image = inpaint_input_image
        self.inpaint_additional_prompt = inpaint_additional_prompt
        self.image_prompts = image_prompts
        self.save_extension = save_extension
        self.require_base64 = require_base64

        if advanced_params is None:
            disable_preview = False
            adm_scaler_positive = 1.5
            adm_scaler_negative = 0.8
            adm_scaler_end = 0.3
            adaptive_cfg = 7.0
            sampler_name = default_sampler
            scheduler_name = default_scheduler
            generate_image_grid = False
            overwrite_step = -1
            overwrite_switch = -1
            overwrite_width = -1
            overwrite_height = -1
            overwrite_vary_strength = -1
            overwrite_upscale_strength = -1
            mixing_image_prompt_and_vary_upscale = False
            mixing_image_prompt_and_inpaint = False
            debugging_cn_preprocessor = False
            skipping_cn_preprocessor = False
            controlnet_softness = 0.25
            canny_low_threshold = 64
            canny_high_threshold = 128
            refiner_swap_method = 'joint'
            freeu_enabled = False
            freeu_b1, freeu_b2, freeu_s1, freeu_s2 = [None] * 4
            debugging_inpaint_preprocessor = False
            inpaint_disable_initial_latent = False
            inpaint_engine = default_inpaint_engine_version
            inpaint_strength = 1.0
            inpaint_respective_field = 0.618
            inpaint_mask_upload_checkbox = False
            invert_mask_checkbox = False
            inpaint_erode_or_dilate = 0

            # Auto set mixing_image_prompt_and_inpaint to True
            if len(self.image_prompts) > 0 and inpaint_input_image is not None:
                print('Mixing Image Prompts and Inpaint Enabled')
                mixing_image_prompt_and_inpaint = True
            if len(self.image_prompts) > 0 and uov_input_image is not None:
                print('Mixing Image Prompts and Vary Upscale Enabled')
                mixing_image_prompt_and_vary_upscale = True

            self.advanced_params = [
                disable_preview, adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, sampler_name, \
                scheduler_name, generate_image_grid, overwrite_step, overwrite_switch, overwrite_width,
                overwrite_height, \
                overwrite_vary_strength, overwrite_upscale_strength, \
                mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint, \
                debugging_cn_preprocessor, skipping_cn_preprocessor, controlnet_softness, canny_low_threshold,
                canny_high_threshold, \
                refiner_swap_method, \
                freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2, \
                debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine, inpaint_strength,
                inpaint_respective_field, \
                inpaint_mask_upload_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate
            ]
        else:
            self.advanced_params = advanced_params
