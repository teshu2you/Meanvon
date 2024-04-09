# coding=utf-8
import os
import re


def alter(to_fixed_file, source_string, new_string):
    with open(to_fixed_file, "r", encoding="utf-8") as f1, open("%s.bak" % to_fixed_file, "w", encoding="utf-8") as f2:
        for line in f1:
            f2.write(re.sub(source_string, new_string, line))
        os.remove(to_fixed_file)
        os.rename("%s.bak" % to_fixed_file, to_fixed_file)


if __name__ == '__main__':
    file = os.path.abspath("../extensions/sadtalker/src/utils/face_enhancer.py")
    ss = "model_path = os.path.join('models/gfpgan/weights', model_name + '.pth')"
    ns = "model_path = os.path.join('../../../../models/gfpgan/weights', model_name + '.pth')"
    alter(to_fixed_file=file, source_string=ss, new_string=ns)

    # file = os.path.abspath("../extensions/sadtalker/src/face3d/extract_kp_videos_safe.py")
    # ss = "root_path = 'models/gfpgan/weights'"
    # ns = "root_path = 'models/gfpgan/weights'"
    # alter(to_fixed_file=file, source_string=ss, new_string=ns)
