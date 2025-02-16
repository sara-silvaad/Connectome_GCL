import subprocess
import os


def run_main_with_model(main_script_path_pre, data_aug, main_script_path_ft):

    os.environ["DATA_AUG"] = data_aug

    # Execute the main.py script
    subprocess.run(["python", main_script_path_pre], check=True)
    subprocess.run(["python", main_script_path_ft], check=True)


augmentations = [ "NoDA", "NoDA_Decoder", "featureMasking_edgeDropping_Decoder", "featureMasking_edgeDropping"]

main_script_path_pre = "main_pre_sup.py"
main_script_path_ft = "main_ft_sup.py"

for data_aug in augmentations:
    print(f"Running main.py with data_aug={data_aug}")
    run_main_with_model(
        main_script_path_pre, data_aug, main_script_path_ft=main_script_path_ft
    )
    print(f"Finished running main.py with data_aug={data_aug}")
