import os
# from modules.sdxl_styles import style_keys
import json


class Test:

    # def get_done_styles(self):
    #     done_styles = []
    #     path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "sdxl_styles_samples")
    #     print(f"path: {path}")
    #     for j, j, k in os.walk(path):
    #         for hh in k:
    #             # print(f"hh: {hh}")
    #             hh = hh.split(".png")[0]
    #             done_styles.append(hh)
    #     print(f"done_styles: {len(done_styles)} - {done_styles}")
    #     return done_styles
    #
    # def get_all_styles(self):
    #     print(f"style_keys: {len(style_keys)} - {style_keys}")
    #     return style_keys
    #
    # def get_undo_style(self, all_styles, done_styles):
    #     undo_styles = list(set(all_styles) - set(done_styles))
    #     print(f"undo_styles: {len(undo_styles)} - {undo_styles}")

    def match_item(self, x):
        _new_out = []
        if isinstance(x, list):
            y = x[0]
            for k in y.items():
                print(k)
                if k[0] == "pooled_output":
                    _new_out.append({k[0]: k[1]})
                else:
                    _new_out.append(json.dumps({k[0]: k[1]}))
            print(_new_out)
        return _new_out

    def csv2txt(self):
        import pandas as pd
        csv_file = "wclfqyqzdyl.csv"
        txt_file = ".txt"
        df = pd.read_csv(csv_file, on_bad_lines="skip")
        for line in df.values:
            with open(str(line[2]) + txt_file, "a+") as f:
                content = str(line[4]).replace("nan", "\n")
                f.writelines(content)


if __name__ == "__main__":
    t = Test()
    # # t.get_undo_style(all_styles=t.get_all_styles(), done_styles=t.get_done_styles())
    # a = [{'model_conds': {'c_crossattn': 3456346, 'y': 43534543534}, 'pooled_output': 77777777777777777777777777}]
    # b = t.match_item(a)
    t.csv2txt()
