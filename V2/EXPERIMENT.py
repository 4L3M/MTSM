import os
import subprocess

selected_scripts = [
    "cnn3.py",
    "cnn4.py",
    "adagrad.py",
    "adam.py",
    "augmentation.py",
    "architectural.py",
    "dropout.py",
    "denseNet21.py",
    "denseNet21-tl.py",
    "denseNet21-ft.py",
    "dropout.py",
    "hyperparameter_tuning-001.py",
    "transfer_learning.py",
]

scripts_folder = os.path.join("V2")

for script in selected_scripts:
    script_path = os.path.join(scripts_folder, script)
    if os.path.exists(script_path):
        print(f"\n🟡 Uruchamianie: {script} ...\n{'='*60}")
        process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        for line in process.stdout:
            print(line, end="")  # Wyświetl linię bez podwójnego \n

        process.wait()
        print(f"\n✅ Zakończono: {script}\n{'='*60}")
    else:
        print(f"❌ Plik nie istnieje: {script_path}")
