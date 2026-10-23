"""Kaggle entrypoint: paste into a notebook cell (or run as a script) with GPU + Internet on.

Expects the XFUND German data attached as a Kaggle Dataset (see train/README.md).
"""
import glob
import os
import subprocess

BRANCH = "kaggle-latin-de-rec-finetune"
REPO = "https://github.com/s4ied/PaddleOCRM.git"
WORK = "/kaggle/working/PaddleOCRM"


def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


# de.train.json lives in the dataset folder that contains de.train/ and de.val/
hits = glob.glob("/kaggle/input/**/de.train.json", recursive=True)
assert hits, "Attach the XFUND-de dataset (needs de.train.json, de.train/, de.val.json, de.val/)"
src = os.path.dirname(hits[0])

if not os.path.isdir(WORK):
    sh(f"git clone --depth 1 -b {BRANCH} {REPO} {WORK}")
else:
    sh("git pull", cwd=WORK)

sh("pip install -q paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/")
sh("pip install -q -r requirements.txt", cwd=WORK)
sh("python -c \"import paddle; paddle.utils.run_check()\"", cwd=WORK)

sh(f"bash train/run.sh {src}", cwd=WORK)
sh("ls output/latin_de_rec_infer && cd output && zip -qr /kaggle/working/latin_de_rec_model.zip .", cwd=WORK)
