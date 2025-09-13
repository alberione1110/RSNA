# basic/evaluate.py
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "_shared"))
from eval import evaluate_experiment

DATA_DIR = "../rsna-pneumonia-detection-challenge/train_png"
CSV_PATH = "../rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
CKPT     = "best_model_basic.pth"   # 해당 폴더의 학습 산출물
MODE     = "basic"

if __name__ == "__main__":
    evaluate_experiment(DATA_DIR, CSV_PATH, CKPT, MODE)
