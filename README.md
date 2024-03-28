# Meanvon
meavon is a self-hosted integrated AI tool that can generate all kinds of pictures and videos. You can run it on your own computer, not to need to rely on high-perform GPU cloud servers. just start with 4GB GPU RAM. 

Based on the following libs --
# Github
📚 Fooocus 📚 Fooocus-MRE 📚 Fooocus-API 📚 RuinedFooocus
📚 stable-diffusion-webui 📚 stable-diffusion-webui-forge
📚 ComfyUI
📚 biniou

# Updates
- 2024.03.27
- -  init version

# Prerequisites
- Minimal hardware :
  - - 64bit CPU
  - - 8GB RAM
  - - Storage requirements :
    - - - for GNU/Linux : at least 20GB for installation without models.
    - - - for Windows : at least 30GB for installation without models.
    - - - for macOS : at least ??GB for installation without models.
     - - - Storage type : HDD

- Operating system :
- - Ubuntu 22.04.3
- - Linux Mint 21.2
- - Windows 10 22H2
- - Windows 11 22H2


# Installation
- Windows 10 / Windows 11
Windows installation has more prerequisites than GNU/Linux one, and requires following softwares (which will be installed automatically) :
- Git
- Python
- OpenSSL
- Visual Studio Build tools
- Windows 10/11 SDK
- ffmpeg

# CUDA support
- install the corresponding Pytorch version with match your nvidia GPU driver version.
```pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 xformers --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 xformers --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 xformers --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 xformers --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 xformers --index-url https://download.pytorch.org/whl/cu118
```
# Features
Primary features:

- Text generation
- Image generation and modification
- Audio generation
- Video generation and modification

other features:
- Cross platform : GNU/Linux, Windows 10/11
- Support for Stable Diffusion SD-1.5, SDXL, SDXL-Turbo, LCM, through built-in model list or standalone .safetensors files
- Support for LoRA models
- Customizable styles through a user-friendly configuration edition.
- Ability to switch between slider and number input modes, allowing users to enter values manually without drag the slider.


# Credits
This application uses the following softwares and technologies :
🤗 Huggingface : Diffusers and Transformers libraries and almost all the generatives models.
* Gradio : webUI
* llama-cpp-python : python bindings for llama-cpp
* Llava
* nllb translation : language translation
* Stable Diffusion : txt2img, img2img, Image variation, inpaint, ControlNet, Text2Video-Zero, img2vid
* Insight Face : faceswapping
* Real ESRGAN : upscaler
* GFPGAN : face restoration
* musicgen melody
* MusicLDM : MusicLDM
* Bark : text2speech
* AnimateLCM : txt2vid

# BUG Fix
- fix1:
ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
```
from scipy import special
from scipy.stats import multivariate_normal
from torchvision.transforms._functional_tensor import rgb_to_grayscale
```

- fix2:
sing xformers cross attention
X:\python_project\Meanvon\venv\lib\site-packages\diffusers\utils\outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(

