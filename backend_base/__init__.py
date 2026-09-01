import os
import importlib.util
import backend_base

__all__ = ['models_info', 'comfyclient_pipeline', 'params_mapper', 'config', 'comfyd']

def get_torch_xformers_cuda_version():
    torch_ver = ""
    cuda_ver = ""
    xformers_ver = ""
    try:
        torch_spec = importlib.util.find_spec("torch")
        for folder in torch_spec.submodule_search_locations:
            ver_file = os.path.join(folder, "version.py")
            if os.path.isfile(ver_file):
                spec = importlib.util.spec_from_file_location("torch_version_import", ver_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                torch_ver = module.__version__
                cuda_ver = getattr(module, 'cuda', "")
        xformers_spec = importlib.util.find_spec("xformers")
        for folder in xformers_spec.submodule_search_locations:
            ver_file = os.path.join(folder, "version.py")
            if os.path.isfile(ver_file):
                spec = importlib.util.spec_from_file_location("xformers_version_import", ver_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                xformers_ver = module.__version__
    except:
        pass
    return torch_ver, xformers_ver, cuda_ver


torch_version, xformers_version, cuda_version = get_torch_xformers_cuda_version()
