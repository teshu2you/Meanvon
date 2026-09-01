import os
import importlib
import importlib.util
import shutil
import subprocess
import sys
import re
import logging
import importlib.metadata
import packaging.version
from packaging.requirements import Requirement

logging.getLogger("torch.distributed.nn").setLevel(logging.ERROR)  # sshh...
logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

re_requirement = re.compile(r"\s*([-\w]+)\s*(?:==\s*([-+.\w]+))?\s*")

python = sys.executable
default_command_live = (os.environ.get('LAUNCH_LIVE_OUTPUT') == "1")
index_url = os.environ.get('INDEX_URL', "")

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def run(command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": 'utf8',
        "errors": 'ignore',
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return (result.stdout or "")


def run_pip(command, desc=None, live=default_command_live):
    try:
        index_url_line = f' --index-url {index_url}' if index_url != '' else ''
        return run(f'"{python}" -m pip {command} --prefer-binary{index_url_line}', desc=f"Installing {desc}",
                   errdesc=f"Couldn't install {desc}", live=live)
    except Exception as e:
        print(e)
        print(f'CMD Failed {desc}: {command}')
        return None


def requirements_met(requirements_file):
    with open(requirements_file, "r", encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if line == "" or line.startswith('#'):
                continue

            requirement = Requirement(line)
            package = requirement.name

            try:
                version_installed = importlib.metadata.version(package)
                installed_version = packaging.version.parse(version_installed)

                # Check if the installed version satisfies the requirement
                if installed_version not in requirement.specifier:
                    print(
                        f"Version mismatch for {package}: Installed version {version_installed} does not meet requirement {requirement}")
                    return False
            except Exception as e:
                print(f"Error checking version for {package}: {e}")
                return False

    return True

def delete_folder_content(folder, prefix=None):
    result = True

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'{prefix}Failed to delete {file_path}. Reason: {e}')
            result = False

    return result



def is_installed_version(package, version_required):
    try:
        version_installed = importlib.metadata.version(package)
    except Exception:
        print()
        print(f'Installing the required version of {package}: {version_required}')
        return False
    if packaging.version.parse(version_required) != packaging.version.parse(version_installed):
        print()
        print(f'The current version of {package} is: {version_installed}. Installing the required version: {version_required}')
        return False
    return True


def verify_installed_version(package_name, package_ver, dependencies = False, use_index = '', package_url = ''):
    result = True
    index_url_line = f' --index-url {use_index}' if use_index != '' else ''
    if package_url:
        package_line = package_url
    else:
        package_line = f'{package_name}=={package_ver}'
    if not is_installed_version(package_name, package_ver):
        run(f'"{python}" -m pip uninstall -y {package_name}')
        if dependencies:
            result = run_pip(f"install -U -I {package_line} {index_url_line} --no-warn-script-location", {package_name}, live=True)
        else:
            result = run_pip(f"install -U -I --no-deps {package_line} {index_url_line} --no-warn-script-location", {package_name}, live=True)
    return result
