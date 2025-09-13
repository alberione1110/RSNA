import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "_shared"))
from eval import evaluate_experiment

DATA_DIR = "../rsna-pneumonia-detection-challenge/train_png"
CSV_PATH = "../rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
CKPT     = "best_model_chaotic_partShuffle.pth"
MODE     = "chaotic_partShuffle"
KEY      = "my_secret_key"
BLOCK    = 8

if __name__ == "__main__":
    evaluate_experiment(DATA_DIR, CSV_PATH, CKPT, MODE, key=KEY, block=BLOCK)
