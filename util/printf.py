import os
import sys
import time
import modules
import inspect


class printF:
    """
        print information in custom method
    """

    def __init__(self, name="None", info=""):
        self.name = name[0] if isinstance(name, list) else name
        self.info = info

    def printf(self):
        time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("[{0:<45}] - {1} : {2}".format(self.name, time_string, self.info))


class MasterName:
    @classmethod
    def get_master_name(cls):
        frame_records = inspect.stack()[1]
        frame = frame_records[0]
        info = inspect.getframeinfo(frame)
        master_name = r"{:<15} -> {}".format(os.path.basename(info.filename), info.function)
        return master_name
