import json
import re

from math import gcd
from os.path import exists


# https://arxiv.org/abs/2307.01952, Appendix I + symmetry
DEFAULT_RESOLUTIONS_LIST = [
        '704*1408', '704*1344', '768*1344', '768*1280', '832*1216', '832*1152',
        '896*1152', '896*1088', '960*1088', '960*1024', '1024*1024', '1024*960',
        '1088*960', '1088*896', '1152*896', '1152*832', '1216*832', '1280*768',
        '1344*768', '1344*704', '1408*704', '1472*704', '1536*640', '1600*640',
        '1664*576', '1728*576'
    # "512×2048",
    # "512×1984",
    # "512×1920",
    # "512×1856",
    # "576×1792",
    # "576×1728",
    # "576×1664",
    # "640×1600",
    # "640×1536",
    # "704×1472",
    # "704×1408",
    # "704×1344",
    # "768×1344",
    # "768×1280",
    # "832×1216",
    # "832×1152",
    # "896×1152",
    # "896×1088",
    # "960×1088",
    # "960×1024",
    # "1024×1024",
    # "1024×960",
    # "1088×960",
    # "1088×896",
    # "1152×896",
    # "1152×832",
    # "1216×832",
    # "1280×768",
    # "1344×768",
    # "1344×704",
    # "1408×704",
    # "1472×704",
    # "1536×640",
    # "1600×640",
    # "1664×576",
    # "1728×576",
    # "1792×576",
    # "1856×512",
    # "1920×512",
    # "1984×512",
    # "2048×512"
]


def load_resolutions(filename=None):
    if filename != None and exists(filename):
        with open(filename) as resolutions_file:
            try:
                resolutions_obj = json.load(resolutions_file)
                if isinstance(resolutions_obj, list) and len(resolutions_obj) > 0:
                    resolutions_dict = get_resolutions_dict(resolutions_obj)
                else:
                    resolutions_dict = get_resolutions_dict(DEFAULT_RESOLUTIONS_LIST)
            except Exception as e:
                print('load_resolutions, e: ' + str(e))
                resolutions_dict = get_resolutions_dict(DEFAULT_RESOLUTIONS_LIST)
            finally:
                resolutions_file.close()
    else:
        resolutions_dict = get_resolutions_dict(DEFAULT_RESOLUTIONS_LIST)
    return resolutions_dict


def string_to_dimensions(resolution_string):
    return list(map(lambda x: int(x), re.findall('\d+', resolution_string)[:2]))


def get_resolution_string(width, height):
    # 1152×896 ∣ 9: 7
    _gcd = gcd(width, height)
    return f'{width}×{height} ∣ ({width//_gcd}:{height//_gcd})'

def get_resolution_new_string(width, height):
    import math
    g = math.gcd(width, height)
    return f'{width}×{height} <span style="color: grey;"> \U00002223 {width // g}:{height // g}</span>'

def annotate_resolution_string(resolution_string):
    width, height = string_to_dimensions(resolution_string)
    return get_resolution_string(width, height)


def get_resolutions_dict(resolutions_list):
    resolutions_dict = {}
    for resolution_string in resolutions_list:
        width, height = string_to_dimensions(resolution_string)
        full_resolution_string = get_resolution_string(width, height)
        resolutions_dict[full_resolution_string] = (width, height)
    return resolutions_dict


resolutions = load_resolutions('resolutions.json')