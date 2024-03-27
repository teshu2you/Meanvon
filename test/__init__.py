import os
from modules.sdxl_styles import style_keys


class Test:

    def get_done_styles(self):
        done_styles = []
        path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "sdxl_styles_samples")
        print(f"path: {path}")
        for j, j, k in os.walk(path):
            for hh in k:
                # print(f"hh: {hh}")
                hh = hh.split(".png")[0]
                done_styles.append(hh)
        print(f"done_styles: {len(done_styles)} - {done_styles}")
        return done_styles

    def get_all_styles(self):
        print(f"style_keys: {len(style_keys)} - {style_keys}")
        return style_keys

    def get_undo_style(self, all_styles, done_styles):
        undo_styles = list(set(all_styles) - set(done_styles))
        print(f"undo_styles: {len(undo_styles)} - {undo_styles}")


if __name__ == "__main__":
    t = Test()
    t.get_undo_style(all_styles=t.get_all_styles(), done_styles=t.get_done_styles())
