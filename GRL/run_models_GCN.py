import subprocess
import os


def run_main_with_model(main_script_path, model):
    os.environ["MODEL"] = model
    subprocess.run(["python", main_script_path], check=True)


main_script_path = "main_GCN.py"

models = ["EncoderDecoderSCFC", "EncoderClassifierSC"]

for model in models:
    print(f"Running main.py with model={model}")
    run_main_with_model(main_script_path, model)
    print(f"Finished running main.py with model={model}")
