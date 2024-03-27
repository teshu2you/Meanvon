import json
import modules
from modules.resolutions import get_resolution_string
from os.path import exists


def load_settings():
    settings = {}
    settings['advanced_mode'] = False
    settings['image_number'] = 1
    settings['save_metadata_json'] = True
    settings['save_metadata_image'] = True
    settings['output_format'] = 'png'
    settings['seed_random'] = True
    settings['same_seed_for_all'] = False
    settings['seed'] = 0
    settings['styles'] = ['Default (Slightly Cinematic)']
    settings['prompt_expansion'] = True
    settings['prompt'] = ''
    settings['negative_prompt'] = ''
    settings['performance'] = 'Speed'
    settings['fixed_steps'] = 30
    settings['custom_steps'] = 24
    settings['custom_switch'] = 0.75
    settings['img2img_mode'] = False
    settings['img2img_start_step'] = 0.06
    settings['img2img_denoise'] = 0.94
    settings['img2img_scale'] = 1.0
    settings['control_lora_canny'] = False
    settings['canny_edge_low'] = 0.2
    settings['canny_edge_high'] = 0.8
    settings['canny_start'] = 0.0
    settings['canny_stop'] = 0.4
    settings['canny_strength'] = 0.8
    settings['canny_model'] = modules.config.default_controlnet_canny_name
    settings['control_lora_depth'] = False
    settings['depth_start'] = 0.0
    settings['depth_stop'] = 0.4
    settings['depth_strength'] = 0.8
    settings['depth_model'] = modules.config.default_controlnet_depth_name
    settings['keep_input_names'] = False
    settings['revision_mode'] = False
    settings['positive_prompt_strength'] = 1.0
    settings['negative_prompt_strength'] = 1.0
    settings['revision_strength_1'] = 1.0
    settings['revision_strength_2'] = 1.0
    settings['revision_strength_3'] = 1.0
    settings['revision_strength_4'] = 1.0
    settings['resolution'] = get_resolution_string(1024, 1024)
    settings['sampler'] = 'dpmpp_2m_sde_gpu'
    settings['scheduler'] = 'karras'
    settings['cfg'] = 7.0
    settings['base_clip_skip'] = -2
    settings['refiner_clip_skip'] = -2
    settings['sharpness'] = 2.0
    settings['base_model'] = modules.config.default_base_model_name
    settings['refiner_model'] = modules.config.default_refiner_model_name
    settings['lora_1_model'] = modules.config.default_loras[0][0]
    settings['lora_1_weight'] = modules.config.default_loras[0][1]
    settings['lora_2_model'] = modules.config.default_loras[1][0]
    settings['lora_2_weight'] = modules.config.default_loras[1][1]
    settings['lora_3_model'] = modules.config.default_loras[2][0]
    settings['lora_3_weight'] = modules.config.default_loras[2][1]
    settings['lora_4_model'] = modules.config.default_loras[3][0]
    settings['lora_4_weight'] = modules.config.default_loras[3][1]
    settings['lora_5_model'] = modules.config.default_loras[4][0]
    settings['lora_5_weight'] = modules.config.default_loras[4][1]
    settings['freeu'] = False
    settings['freeu_b1'] = 1.01
    settings['freeu_b2'] = 1.02
    settings['freeu_s1'] = 0.99
    settings['freeu_s2'] = 0.95

    if exists('settings.json'):
        with open('settings.json') as settings_file:
            try:
                settings_obj = json.load(settings_file)
                for k in settings.keys():
                    if k in settings_obj:
                        settings[k] = settings_obj[k]
            except Exception as e:
                print('load_settings, e: ' + str(e))
            finally:
                settings_file.close()

    return settings


default_settings = load_settings()


# 首次运行需要从Hugging Face下载ClIP模型等 也需要设置代理
# 设置自己的 http代理
# This config about proxy only for China
# os.environ['http_proxy'] = 'http://127.0.0.1:7890/'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890/'

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 指定英伟达显卡，序号从0开始
# Specify the NVIDIA GPU to use，the serial number starts from 0


# gradio_args = dict(
#     share=True,
#     # 产生一个gradio 的分享链接，可以通过gradio的网站进行访问
#     # generate a gradio share link, you can access it through the gradio website
#     auth=dict(
#         # username='root',
#         # password='123456',
#         message='Place enter your username and password; 请输入用户名何密码'
#     ),
#     # 设置用户名何密码， 如果你不想设置请注释掉
#     # set username and password， if you don't want to set it, please comment it out
#     head_html="""
#             <div style="text-align: center;line-height:0">
#                 <h1>Stable Video Diffusion WebUI</h1>
#                 <p>Upload an image to create a Video with the image.</p>
#             </div>
#             """,
#     # 设置gradio的头部信息
#     # Set the header information of gradio
#     show_api=True,
#     # 显示api 信息 show api information
# )

auto_adjust_img = dict(
    min_width=256,  # 图片最小宽度 Image minimum width
    min_height=256,  # 图片最小高度 Image minimum height
    max_height=1024,  # 图片最大宽度 Image maximum width
    max_width=1024,  # 图片最大高度 Image maximum height
    multiple_of_N=64  # 图片的宽高必须是N的倍数 The width and height of the image must be a multiple of N
)
# 自动调整图片分辨率,自动调整到符合要求的分辨率
# Automatically adjust the image resolution, automatically adjust to the resolution that meets the requirements

img_resize_to_HW = dict(
    target_width=1024,  # 目标宽度
    target_height=576,  # 目标高度
)
# 因为往往在训练尺寸下的图片尺寸能达到比较好的效果,但是硬剪裁会扭曲图片，所以使用从中心剪裁
# Because often, using images at training size can achieve better
# results, but hard cropping can distort the images, so we use center cropping.

creat_video_by_opencv = False
# 使用opencv生成视频, 但是发现会有一些编码的问题，所以默认关闭，默认使用moviepy
# Use opencv to generate video, but it is found that there will be some encoding problems,
# so it is turned off by default,default use moviepy

infer_args = dict(
    default_fps=6,
    # 默认的视频帧率
    # Default video fps
    nsfw_filter_checkbox=dict(
        enable=True,
        default=True
    )
    # 是否可以跳过nsfw过滤, 默认打开此选择,
    # 简而言之小心别人用你部署的服务推理不良内容，建议enable=False 关掉这个选项 并且设置default=False
    # Whether nsfw filtering can be skipped
)
# 推理参数
# Inference parameters