import subprocess
import sys
sys.stdout.reconfigure(encoding='utf-8')



scripts = [
    "cnn3.py",
    "adagrad.py",
  #  "deepGACNN.py",
    "dropout.py",
   # "GACNN2.py",
    "gpu.py",
#    "gridCNN.py",
 #   "OptunaCNN2.py",
    "pruning.py"
]

with open("console_output_log.txt", "w", encoding="utf-8") as logfile:
    for script in scripts:
        print(f"\n🚀 Uruchamianie: {script}\n{'='*50}")
        logfile.write(f"\n🚀 Uruchamianie: {script}\n{'='*50}\n")

        process = subprocess.Popen(
            ["python", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end='')           # Na żywo do konsoli
            logfile.write(line)           # I do pliku

        process.wait()
        logfile.write(f"\n✅ Zakończono: {script}\n")
        print(f"\n✅ Zakończono: {script}\n{'='*50}")
