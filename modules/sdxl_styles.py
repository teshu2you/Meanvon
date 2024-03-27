import math
import os
import re
import json
import re
import random
import sys
import modules.config
from modules.util import get_files_from_folder
from os.path import exists
from modules.util import join_prompts
from util.printf import printF, MasterName

# cannot use modules.config - validators causing circular imports
styles_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sdxl_styles/'))
wildcards_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../wildcards/'))
wildcards_max_bfs_depth = 64

base_styles = [
    {
        "name": "Default (Slightly Cinematic)",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
    }
]

default_styles_files = ['sdxl_styles_fooocus.json', 'sdxl_styles_sai.json', 'sdxl_styles_twri.json',
                        'sdxl_styles_diva.json', 'sdxl_styles_mre.json']

hot_style_keys = {
    "sai": ["sai-3d-model", "sai-analog film", "sai-anime",
            "sai-cinematic", "sai-comic book", "sai-craft clay",
            "sai-digital art", "sai-enhance", "sai-fantasy art",
            "sai-isometric", "sai-line art", "sai-lowpoly",
            "sai-neonpunk", "sai-origami", "sai-photographic",
            "sai-pixel art", "sai-texture"],
    "artstyle": [
        "artstyle-abstract", "artstyle-abstract expressionism", "artstyle-art deco",
        "artstyle-art nouveau", "artstyle-constructivist", "artstyle-cubist",
        "artstyle-expressionist", "artstyle-graffiti", "artstyle-hyperrealism", "artstyle-impressionist",
        "artstyle-pointillism", "artstyle-pop art", "artstyle-psychedelic",
        "artstyle-renaissance", "artstyle-steampunk", "artstyle-surrealist",
        "artstyle-typography", "artstyle-watercolor"
    ],
    "futuristic": [
        "futuristic-biomechanical", "futuristic-biomechanical cyberpunk", "futuristic-cybernetic",
        "futuristic-cybernetic robot", "futuristic-cyberpunk cityscape", "futuristic-futuristic",
        "futuristic-retro cyberpunk", "futuristic-retro futurism", "futuristic-sci-fi",
        "futuristic-vaporwave", "Neo-Futurism"
    ],
    "photo": ["photo-alien", "photo-film noir", "photo-hdr", "photo-long exposure", "photo-neon noir",
              "photo-silhouette", "photo-tilt-shift", "Faded Polaroid Photo"
              ],
    "ads": ["ads-advertising", "ads-automotive", "ads-corporate",
            "ads-fashion editorial", "ads-food photography",
            "ads-luxury", "ads-real estate", "ads-retail",
            ],
    "cinematic": ["cinematic-diva", "sai-analog film",
                  ],
    "Fantasy": [
        "Dark Fantasy",
    ],
    "Ink": [
        "Ink Dripping Drawing", "Japanese Ink Drawing",
    ],
    "Logo": [
        "Logo Design",
    ],
    "Fashion": [
        "ads-fashion editorial", "ads-automotive",
    ],
    "photography": [
        "ads-food photography", "sai-photographic",
    ],
    "cyberpunk": [
        "futuristic-biomechanical cyberpunk", "futuristic-biomechanical",
    ],
    "portrait": [
        "sai-3d-model", "sai-analog film", "sai-cinematic", "sai-photographic"
    ]
}


def normalize_key(k):
    k = k.replace('-', ' ')
    words = k.split(' ')
    words = [w[:1].upper() + w[1:].lower() for w in words]
    k = ' '.join(words)
    k = k.replace('3d', '3D')
    k = k.replace('Sai', 'SAI')
    k = k.replace('Mre', 'MRE')
    k = k.replace('(s', '(S')
    return k


styles = {}
styles_files = get_files_from_folder(styles_path, ['.json'])

for x in ['sdxl_styles_fooocus.json',
          'sdxl_styles_sai.json',
          'sdxl_styles_mre.json',
          'sdxl_styles_twri.json',
          'sdxl_styles_diva.json',
          'sdxl_styles_marc_k3nt3l.json']:
    if x in styles_files:
        styles_files.remove(x)
        styles_files.append(x)

for styles_file in styles_files:
    try:
        with open(os.path.join(styles_path, styles_file), encoding='utf-8') as f:
            for entry in json.load(f):
                name = normalize_key(entry['name'])
                prompt = entry['prompt'] if 'prompt' in entry else ''
                negative_prompt = entry['negative_prompt'] if 'negative_prompt' in entry else ''
                styles[name] = (prompt, negative_prompt)
    except Exception as e:
        print(str(e))
        print(f'Failed to load style file {styles_file}')


def migrate_style_from_v1(style):
    if style == 'cinematic-default':
        return ['Default (Slightly Cinematic)']
    elif style == 'None':
        return []
    else:
        return [normalize_key(style)]


def styles_list_to_styles_dict(styles_list=None, base_dict=None):
    styles_dict = {} if base_dict is None else base_dict
    if isinstance(styles_list, list) and len(styles_list) > 0:
        for entry in styles_list:
            name, prompt, negative_prompt = normalize_key(entry['name']), \
                normalize_key(entry['prompt'] if 'prompt' in entry else ''), \
                normalize_key(entry['negative_prompt'] if 'negative_prompt' in entry else '')
            if name not in styles_dict:
                styles_dict |= {name: (prompt, negative_prompt)}
    return styles_dict


def load_styles(filename=None, base_dict=None):
    styles_dict = {} if base_dict is None else base_dict
    full_path = os.path.join(styles_path, filename) if filename != None else None
    if full_path is not None and os.path.exists(full_path):
        with open(full_path, encoding='utf-8') as sf:
            try:
                styles_obj = json.load(sf)
                styles_list_to_styles_dict(styles_obj, styles_dict)
            except Exception as e:
                printF(name=MasterName.get_master_name(), info="load_styles, e: {}".format(e)).printf()
            finally:
                sf.close()
    return styles_dict


styles = styles_list_to_styles_dict(base_styles)
for styles_file in default_styles_files:
    styles = load_styles(styles_file, styles)

all_styles_files = get_files_from_folder(styles_path, ['.json'])
for styles_file in all_styles_files:
    if styles_file not in default_styles_files:
        styles = load_styles(styles_file, styles)

style_keys = list(styles.keys())

fooocus_expansion = "Fooocus V2"
default_style = "Default (Slightly Cinematic)"
legal_style_names = [fooocus_expansion] + style_keys
default_legal_style_names = [fooocus_expansion] + ['Fooocus Enhance', 'Fooocus Photograph', 'Fooocus Cinematic']

SD_XL_BASE_RATIOS = {
    "0.5": (704, 1408),
    "0.52": (704, 1344),
    "0.57": (768, 1344),
    "0.6": (768, 1280),
    "0.68": (832, 1216),
    "0.72": (832, 1152),
    "0.78": (896, 1152),
    "0.82": (896, 1088),
    "0.88": (960, 1088),
    "0.94": (960, 1024),
    "1.0": (1024, 1024),
    "1.07": (1024, 960),
    "1.13": (1088, 960),
    "1.21": (1088, 896),
    "1.29": (1152, 896),
    "1.38": (1152, 832),
    "1.46": (1216, 832),
    "1.67": (1280, 768),
    "1.75": (1344, 768),
    "1.91": (1344, 704),
    "2.0": (1408, 704),
    "2.09": (1472, 704),
    "2.4": (1536, 640),
    "2.5": (1600, 640),
    "2.89": (1664, 576),
    "3.0": (1728, 576),
}

aspect_ratios = {}

# import math

for kk, (w, h) in SD_XL_BASE_RATIOS.items():
    txt = f'{w}×{h}'

    # gcd = math.gcd(w, h)
    # txt += f' {w//gcd}:{h//gcd}'

    aspect_ratios[txt] = (w, h)
    if kk == "1.0":
        default_aspect_ratio = txt


def apply_style(style, positive):
    p, n = styles[style]
    return p.replace('{prompt}', positive).splitlines(), n.splitlines()


def apply_wildcards(wildcard_text, rng, i, read_wildcards_in_order):
    for _ in range(wildcards_max_bfs_depth):
        placeholders = re.findall(r'__([\w-]+)__', wildcard_text)
        if len(placeholders) == 0:
            return wildcard_text

        print(f'[Wildcards] processing: {wildcard_text}')
        for placeholder in placeholders:
            try:
                matches = [x for x in modules.config.wildcard_filenames if os.path.splitext(os.path.basename(x))[0] == placeholder]
                words = open(os.path.join(modules.config.path_wildcards, matches[0]), encoding='utf-8').read().splitlines()
                words = [x for x in words if x != '']
                assert len(words) > 0
                if read_wildcards_in_order:
                    wildcard_text = wildcard_text.replace(f'__{placeholder}__', words[i % len(words)], 1)
                else:
                    wildcard_text = wildcard_text.replace(f'__{placeholder}__', rng.choice(words), 1)
            except:
                print(f'[Wildcards] Warning: {placeholder}.txt missing or empty. '
                      f'Using "{placeholder}" as a normal word.')
                wildcard_text = wildcard_text.replace(f'__{placeholder}__', placeholder)
            print(f'[Wildcards] {wildcard_text}')

    print(f'[Wildcards] BFS stack overflow. Current text: {wildcard_text}')
    return wildcard_text


def get_words(arrays, totalMult, index):
    if len(arrays) == 1:
        return [arrays[0].split(',')[index]]
    else:
        words = arrays[0].split(',')
        word = words[index % len(words)]
        index -= index % len(words)
        index /= len(words)
        index = math.floor(index)
        return [word] + get_words(arrays[1:], math.floor(totalMult / len(words)), index)


def apply_arrays(text, index):
    arrays = re.findall(r'\[\[(.*?)\]\]', text)
    if len(arrays) == 0:
        return text

    print(f'[Arrays] processing: {text}')
    mult = 1
    for arr in arrays:
        words = arr.split(',')
        mult *= len(words)

    index %= mult
    chosen_words = get_words(arrays, mult, index)

    i = 0
    for arr in arrays:
        text = text.replace(f'[[{arr}]]', chosen_words[i], 1)
        i = i + 1

    return text
