import os, sys
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), "_shared"))
from gradcam import run_gradcam_for_experiment

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT/"rsna-pneumonia-detection-challenge/stage_2_train_images"
CSV_PATH = ROOT/"rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
CHAOTIC_KEY = "my_secret_key"
BLOCK = 8

EXPS = [
    ("basic",                  "best_model_basic.pth",                  "basic",                  {}),
    ("pixelShuffle-only",      "best_model_pixelShuffle.pth",           "pixelShuffle-only",      {}),
    ("partShuffle-only",       "best_model_partShuffle.pth",            "partShuffle-only",       {"block": BLOCK}),
    ("chaotic-only",           "best_model_chaoticOnly.pth",            "chaotic-only",           {"key": CHAOTIC_KEY}),
    ("chaotic_pixelShuffle",   "best_model_chaotic_pixelShuffle.pth",   "chaotic_pixelShuffle",   {"key": CHAOTIC_KEY}),
    ("chaotic_partShuffle",    "best_model_chaotic_partShuffle.pth",    "chaotic_partShuffle",    {"key": CHAOTIC_KEY, "block": BLOCK}),
]

if __name__ == "__main__":
    for folder, ckpt, mode, extra in EXPS:
        run_gradcam_for_experiment(
            exp_dir=str(ROOT/folder),
            ckpt_path=str(ROOT/folder/ckpt),
            mode=mode,
            data_dir=str(DATA_DIR),
            csv_path=str(CSV_PATH),
            key=extra.get("key"),
            block=extra.get("block", 8),
            image_size=224,
            num_samples=12,
        )
