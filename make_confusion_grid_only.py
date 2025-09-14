# RSNA/export_confusion_grid_only.py
# 6개 케이스의 eval_result.json을 읽어 2x3 혼동행렬 그리드와 공통 colorbar를 저장

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ====== 경로/케이스 정의 ======
ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "_reports"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUTDIR / "confusion_grid.png"

CASES = [
    "Original",
    "Pixel Random",
    "Part Shuffle",
    "Chaotic Only",
    "Chaotic + Pixel Random",
    "Chaotic + Part Shuffle",
]

EVAL_JSONS = {
    "Original":                 ROOT / "basic"                / "eval_result.json",
    "Pixel Random":             ROOT / "pixelShuffle-only"    / "eval_result.json",
    "Part Shuffle":             ROOT / "partShuffle-only"     / "eval_result.json",
    "Chaotic Only":             ROOT / "chaotic-only"         / "eval_result.json",
    "Chaotic + Pixel Random":   ROOT / "chaotic_pixelShuffle" / "eval_result.json",
    "Chaotic + Part Shuffle":   ROOT / "chaotic_partShuffle"  / "eval_result.json",
}

def load_confusion_matrices():
    """각 케이스의 eval_result.json -> 2x2 비율 행렬(dict)."""
    cms = {}
    for title in CASES:
        jpath = EVAL_JSONS.get(title)
        if not jpath or not jpath.exists():
            print(f"⚠️  Skip (json not found): {jpath}")
            continue
        with open(jpath, "r") as f:
            data = json.load(f)
        tn = int(data["metrics"]["tn"])
        fp = int(data["metrics"]["fp"])
        fn = int(data["metrics"]["fn"])
        tp = int(data["metrics"]["tp"])
        total = max(1, tn + fp + fn + tp)
        cms[title] = np.array([[tn, fp], [fn, tp]], dtype=np.float32) / float(total)
    return cms

def draw_confusion_grid(cms: dict, out_path: Path):
    """
    2x3 그리드 + 별도 colorbar 열(겹침 방지).
    """
    # ----- Figure & GridSpec: 마지막 1열은 colorbar 전용 -----
    fig = plt.figure(figsize=(10, 6), constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=4,  # 3열 플롯 + 1열 컬러바
                          width_ratios=[1, 1, 1, 0.05],
                          wspace=0.25, hspace=0.30)

    axes = []
    for r in range(2):
        for c in range(3):
            axes.append(fig.add_subplot(gs[r, c]))

    cax = fig.add_subplot(gs[:, 3])  # colorbar 전용 축

    im_ref = None
    for i, title in enumerate(CASES):
        ax = axes[i]
        cm = cms.get(title)
        if cm is None:
            ax.axis("off")
            continue

        im_ref = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)

        # 값 표기 (가독성 위해 값에 따라 글자색 선택)
        for (r, c), v in np.ndenumerate(cm):
            txt_color = "white" if v >= 0.5 else "black"
            ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                    color=txt_color, fontsize=10, fontweight="bold")

        ax.set_title(title, fontsize=10)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Neg", "Pred: Pos"], fontsize=9)
        ax.set_yticklabels(["True: Neg", "True: Pos"], fontsize=9)

    # 공통 colorbar를 전용 축(cax)에 배치 → 다른 플롯과 절대 겹치지 않음
    if im_ref is not None:
        cbar = fig.colorbar(im_ref, cax=cax)
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label("Proportion", fontsize=10)

    # 플롯 주변 여백 미세 조정
    fig.suptitle("Confusion Matrices (2x3)", fontsize=12, y=0.98)
    fig.subplots_adjust(left=0.06, right=0.95, top=0.92, bottom=0.07)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"✅ Saved {out_path}")

if __name__ == "__main__":
    cms = load_confusion_matrices()
    draw_confusion_grid(cms, OUT_PATH)
