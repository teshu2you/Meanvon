# as in k-diffusion (sampling.py)
AUTH_FILENAME = 'auth.json'
STEPS_SPEED = 30
STEPS_QUALITY = 60
STEPS_EXTREME_SPEED = 8
STEPS_LIGHTNING = 8
STEPS_LCM = 8
STEPS_TURBO = 8
SWITCH_SPEED = 20
SWITCH_QUALITY = 40
STEPS_HYPER_SD = 4

TYPE_LIGHTNING = "lightning"
TYPE_LCM = "lcm"
TYPE_TURBO = "turbo"
# TYPE_HYPER_SD must be set "lower"
TYPE_HYPER_SD = "hyper-sd"
TYPE_HunyuanDiT = "hunyuandit"
TYPE_Flux = "flux"
# extreme_speed disused
TYPE_EXTREME_SPEED = "extreme_speed"
TYPE_NORMAL = "normal"

# import parameter, used by webui_server.py and work.py
MIN_SEED = 0
MAX_SEED = 2**32 - 1

# exclusive, needed by modules\expansion.py -> transformers\trainer_utils.py -> np.random.seed()
SEED_LIMIT_NUMPY = 2**32

# min - native SDXL resolution (1024x1024), max - determined by SDXL context size (2048)
MIN_MEGAPIXELS = 1.0
MAX_MEGAPIXELS = 4.0

# min - native SD 1.5 resolution (512x512), max - determined by SD 2.x context size (1024)
MIN_MEGAPIXELS_SD = 0.25
MAX_MEGAPIXELS_SD = 1.0

