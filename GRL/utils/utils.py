import os
import shutil
import re


def check_path(path_to_save, config_path="./config.py"):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        exp_path = os.path.join(path_to_save, "exp_001")
        os.makedirs(exp_path)
        shutil.copy(config_path, exp_path)
        return exp_path

    existing_exp = [d for d in os.listdir(path_to_save) if re.match(r"exp_\d{3}", d)]

    if not existing_exp:
        exp_path = os.path.join(path_to_save, "exp_001")
        os.makedirs(exp_path)
        shutil.copy(config_path, exp_path)
        return exp_path

    existing_exp.sort()
    last_exp = existing_exp[-1]
    last_exp_num = int(re.search(r"\d{3}", last_exp).group())

    new_exp_num = last_exp_num + 1
    new_exp_str = f"exp_{new_exp_num:03d}"
    exp_path = os.path.join(path_to_save, new_exp_str)
    os.makedirs(exp_path)
    shutil.copy(config_path, exp_path)

    return exp_path
