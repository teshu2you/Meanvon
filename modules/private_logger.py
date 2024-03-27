import os
import sys

import adapter.args_manager
import modules.config
import json
import urllib.parse
from modules.flags import OutputFormat
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from modules.util import generate_temp_filename
from modules.meta_parser import MetadataParser, get_exif
from util.printf import printF, MasterName
from shutil import copy

log_cache = {}


def get_current_html_path(output_format=None):
    output_format = output_format if output_format else modules.config.default_output_format
    date_string, local_temp_filename, only_name = generate_temp_filename(folder=modules.config.path_outputs,
                                                                         extension=output_format)
    html_name = os.path.join(os.path.dirname(local_temp_filename), "Gallery_Images_" + date_string + '.html')
    return html_name


def log(img, metadata, metadata_parser: MetadataParser | None = None, output_format=None, save_metadata_json=True,
        input_image_filename="", keep_input_names="", base_model_name_prefix="") -> str:
    path_outputs = adapter.args_manager.args.temp_path if adapter.args_manager.args.disable_image_log else modules.config.path_outputs
    output_format = output_format if output_format else modules.config.default_output_format

    date_string, local_temp_filename, only_name = generate_temp_filename(folder=path_outputs, extension=output_format,
                                                                         base=input_image_filename if keep_input_names else base_model_name_prefix)
    os.makedirs(os.path.dirname(local_temp_filename), exist_ok=True)

    parsed_parameters = metadata_parser.parse_string(metadata.copy()) if metadata_parser is not None else ''
    image = Image.fromarray(img)

    if metadata is not None:
        metadata_dict = {}

        for i in metadata:
            metadata_dict.update({
                i[1]: str(i[2])
            })
        _metadata = json.dumps(metadata_dict, ensure_ascii=False)
        print(f"_metadata: {_metadata}")

        with open(modules.config.last_prompt_path, 'w', encoding='utf-8') as json_file:
            json_file.write(_metadata)
            json_file.close()

        if save_metadata_json:
            json_path = local_temp_filename.replace(f'.{output_format}', '.json')
            copy(modules.config.last_prompt_path, json_path)

    if output_format == OutputFormat.PNG.value:
        if parsed_parameters != '':
            pnginfo = PngInfo()
            pnginfo.add_text('parameters', parsed_parameters)
            pnginfo.add_text('fooocus_scheme', metadata_parser.get_scheme().value)
        else:
            pnginfo = None
        image.save(local_temp_filename, pnginfo=pnginfo)
    elif output_format == OutputFormat.JPEG.value:
        image.save(local_temp_filename, quality=95, optimize=True, progressive=True, exif=get_exif(parsed_parameters,
                                                                                                   metadata_parser.get_scheme().value) if metadata_parser else Image.Exif())
    elif output_format == OutputFormat.WEBP.value:
        image.save(local_temp_filename, quality=95, lossless=False, exif=get_exif(parsed_parameters,
                                                                                  metadata_parser.get_scheme().value) if metadata_parser else Image.Exif())
    else:
        image.save(local_temp_filename)

    if adapter.args_manager.args.disable_image_log:
        return local_temp_filename

    html_name = os.path.join(os.path.dirname(local_temp_filename), "Gallery_Images_" + date_string + '.html')

    css_styles = (
        "<style>"
        "body { background-color: #121212; color: #E0E0E0; } "
        "a { color: #BB86FC; } "
        ".metadata { border-collapse: collapse; width: 100%; } "
        ".metadata .label { width: 15%; } "
        ".metadata .value { width: 85%; font-weight: bold; } "
        ".metadata th, .metadata td { border: 1px solid #4d4d4d; padding: 4px; } "
        ".image-container img { height: auto; max-width: 512px; display: block; padding-right:10px; } "
        ".image-container div { text-align: center; padding: 4px; } "
        "hr { border-color: gray; } "
        "button { background-color: black; color: white; border: 1px solid grey; border-radius: 5px; padding: 5px 10px; text-align: center; display: inline-block; font-size: 16px; cursor: pointer; }"
        "button:hover {background-color: grey; color: black;}"
        "</style>"
    )

    js = (
        """<script>
        function to_clipboard(txt) { 
        txt = decodeURIComponent(txt);
        if (navigator.clipboard && navigator.permissions) {
            navigator.clipboard.writeText(txt)
        } else {
            const textArea = document.createElement('textArea')
            textArea.value = txt
            textArea.style.width = 0
            textArea.style.position = 'fixed'
            textArea.style.left = '-999px'
            textArea.style.top = '10px'
            textArea.setAttribute('readonly', 'readonly')
            document.body.appendChild(textArea)

            textArea.select()
            document.execCommand('copy')
            document.body.removeChild(textArea)
        }
        alert('Copied to Clipboard!\\nPaste to prompt area to load parameters.\\nCurrent clipboard content is:\\n\\n' + txt);
        }
        </script>"""
    )

    begin_part = f"<!DOCTYPE html><html><head><title>MeanVon Log {date_string}</title>{css_styles}</head><body>{js}<p>MeanVon Log {date_string} (private)</p>\n<p>Metadata is embedded if enabled in the config or developer debug mode. You can find the information for each image in line Metadata Scheme.</p><!--MeanVon-log-split-->\n\n"
    end_part = f'\n<!--MeanVon-log-split--></body></html>'

    middle_part = log_cache.get(html_name, "")

    if middle_part == "":
        if os.path.exists(html_name):
            existing_split = open(html_name, 'r', encoding='utf-8').read().split('<!--MeanVon-log-split-->')
            if len(existing_split) == 3:
                middle_part = existing_split[1]
            else:
                middle_part = existing_split[0]

    div_name = only_name.replace('.', '_')
    item = f"<div id=\"{div_name}\" class=\"image-container\"><hr><table><tr>\n"
    item += f"<td><a href=\"{only_name}\" target=\"_blank\"><img src='{only_name}' onerror=\"this.closest('.image-container').style.display='none';\" loading='lazy'/></a><div>{only_name}</div></td>"
    item += "<td><table class='metadata'>"
    for label, key, value in metadata:
        value_txt = str(value).replace('\n', ' </br> ')
        item += f"<tr><td class='label'>{label}</td><td class='value'>{value_txt}</td></tr>\n"
    item += "</table>"

    js_txt = urllib.parse.quote(json.dumps({k: v for _, k, v in metadata}, indent=0), safe='')
    item += f"</br><button onclick=\"to_clipboard('{js_txt}')\">Copy to Clipboard</button>"

    item += "</td>"
    item += "</tr></table></div>\n\n"

    middle_part = item + middle_part

    with open(html_name, 'w', encoding='utf-8') as f:
        f.write(begin_part + middle_part + end_part)

    printF(name=MasterName.get_master_name(),
           info="[Info] Image generated with private log at: {}".format(html_name)).printf()

    log_cache[html_name] = middle_part

    return local_temp_filename
