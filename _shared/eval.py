# _shared/eval.py
import os, sys, argparse, json, math
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score
)
from tqdm import tqdm

# 내부 모듈
sys.path.append(os.path.dirname(__file__))
from dataset import RSNADataset
from train_util import default_transform
from encrypt_ops import (
    apply_identity, apply_shuffle_only, apply_partshuffle_only,
    apply_xor_only, apply_xor_then_shuffle, apply_xor_then_partshuffle
)

# -------------------------
# 유틸
# -------------------------
def load_label_map(csv_path: str) -> Dict[str, int]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    return {str(r["patientId"]): int(r["Target"]) for _, r in df.iterrows()}

def get_preprocess_fn(mode: str):
    mode = mode.lower()
    if mode in ["basic", "identity", "none"]:
        return apply_identity, {}
    if mode in ["pixel", "pixelshuffle-only", "pixelshuffle"]:
        return apply_shuffle_only, {}
    if mode in ["part", "partshuffle-only", "partshuffle"]:
        return apply_partshuffle_only, {}
    if mode in ["chaotic-only", "xor-only", "chaotic"]:
        return apply_xor_only, {}
    if mode in ["chaotic_pixelshuffle", "xor_then_pixel", "xor_then_shuffle"]:
        return apply_xor_then_shuffle, {}
    if mode in ["chaotic_partshuffle", "xor_then_part", "xor_then_block"]:
        return apply_xor_then_partshuffle, {}
    raise ValueError(f"Unknown mode: {mode}")

def build_model(num_classes=2):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m

def predict(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, preds, gts = [], [], []
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval  ", ncols=100):
            x = x.to(device)
            out = model(x)
            p = softmax(out)[:, 1]
            pr = out.argmax(1)
            probs.append(p.detach().cpu().numpy())
            preds.append(pr.detach().cpu().numpy())
            gts.append(y.numpy())
    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    gts   = np.concatenate(gts)
    return probs, preds, gts

def compute_metrics(gts, preds, probs) -> Dict[str, float]:
    acc = accuracy_score(gts, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(gts, preds, average="binary", zero_division=0)
    cm = confusion_matrix(gts, preds)
    # ROC-AUC / PR-AUC (양성 클래스=1, 확률 필요)
    try:
        roc = roc_auc_score(gts, probs)
    except Exception:
        roc = float("nan")
    try:
        ap = average_precision_score(gts, probs)
    except Exception:
        ap = float("nan")
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "roc_auc": roc, "ap": ap,
        "tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1]),
    }

def mcnemar_test(y_true, y_pred_a, y_pred_b) -> Dict[str, float]:
    """
    동일 검증세트에서 모델 A,B 예측 비교.
    b = A 맞고 B 틀림, c = A 틀리고 B 맞음
    continuity corrected chi^2 = ((|b-c|-1)^2)/(b+c)
    정확 이항 검정 대신 간이 통계 제공.
    """
    # b,c 계산
    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_approx": 1.0}
    chi2 = (abs(b - c) - 1)**2 / (b + c)
    # 근사 p-value (자유도1 카이제곱 상한)
    # 카이제곱 CDF 대신 근사치: p ~ exp(-0.5*chi2) * (1 + chi2/2 + ...) 사용 가능
    # 여기서는 간단히 survival function을 근사
    # 안전하게 상한만 제공
    p_upper = math.exp(-0.5 * chi2)
    return {"b": b, "c": c, "chi2": float(chi2), "p_approx_upper": float(p_upper)}

# -------------------------
# 메인 평가
# -------------------------
def evaluate_experiment(
    data_dir: str,
    csv_path: str,
    checkpoint: str,
    mode: str,
    image_size: int = 224,
    batch_size: int = 32,
    key: Optional[str] = None,
    block: int = 8,
    out_path: Optional[str] = None,
    val_seed: int = 42,
):
    label_map = load_label_map(csv_path)
    ids = sorted(list(label_map.keys()))
    _, val_ids = train_test_split(ids, test_size=0.2, random_state=val_seed)
    tfm = default_transform(image_size)

    preprocess_fn, kwargs = get_preprocess_fn(mode)
    # chaotic / xor 계열이면 key 전달
    if "xor" in preprocess_fn.__name__ or "chaotic" in mode:
        if key is None:
            key = "my_secret_key"
        kwargs["key"] = key
    # partShuffle 계열이면 block 전달
    if "partshuffle" in preprocess_fn.__name__ or "part" in mode:
        kwargs["block"] = block

    val_ds = RSNADataset(
        image_dir=data_dir,
        label_map=label_map,
        transform=tfm,
        filter_ids=val_ids,
        preprocess_fn=preprocess_fn,
        preprocess_kwargs=kwargs,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)

    probs, preds, gts = predict(model, val_loader, device)
    metrics = compute_metrics(gts, preds, probs)

    result = {
        "mode": mode,
        "checkpoint": checkpoint,
        "image_size": image_size,
        "batch_size": batch_size,
        "key": key if key is not None else "",
        "block": block,
        "val_seed": val_seed,
        "metrics": metrics,
    }

    if out_path is None:
        out_path = os.path.join(os.path.dirname(checkpoint), "eval_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"✅ Saved metrics to {out_path}")

    # 예측 저장(비교용)
    np.savez(os.path.join(os.path.dirname(checkpoint), "eval_preds.npz"),
             y_true=gts, y_pred=preds, y_prob=probs)
    print("✅ Saved predictions (eval_preds.npz)")

    return result

def compare_mcnemar(preds_a_path: str, preds_b_path: str, out_path: Optional[str] = None):
    a = np.load(preds_a_path)
    b = np.load(preds_b_path)
    y_true = a["y_true"]
    y_pred_a = a["y_pred"]
    y_pred_b = b["y_pred"]
    assert np.array_equal(y_true, b["y_true"]), "두 결과의 검증 세트가 동일해야 합니다."

    mc = mcnemar_test(y_true, y_pred_a, y_pred_b)
    print(f"McNemar: b={mc['b']}, c={mc['c']}, chi2={mc['chi2']:.4f}, p≈≤{mc['p_approx_upper']:.4f}")

    res = {"mcnemar": mc, "preds_a": preds_a_path, "preds_b": preds_b_path}
    if out_path is None:
        out_path = os.path.join(os.path.dirname(preds_a_path), "mcnemar_result.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"✅ Saved McNemar to {out_path}")
    return res

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="단일 실험 평가")
    pe.add_argument("--data_dir", required=True)
    pe.add_argument("--csv_path", required=True)
    pe.add_argument("--checkpoint", required=True)
    pe.add_argument("--mode", required=True,
                    help="basic | pixelShuffle-only | partShuffle-only | chaotic-only | chaotic_pixelShuffle | chaotic_partShuffle")
    pe.add_argument("--key", default=None)
    pe.add_argument("--block", type=int, default=8)
    pe.add_argument("--image_size", type=int, default=224)
    pe.add_argument("--batch_size", type=int, default=32)
    pe.add_argument("--val_seed", type=int, default=42)
    pe.add_argument("--out", default=None)

    pm = sub.add_parser("mcnemar", help="두 결과의 McNemar 비교")
    pm.add_argument("--preds_a", required=True, help=".../eval_preds.npz (실험 A)")
    pm.add_argument("--preds_b", required=True, help=".../eval_preds.npz (실험 B)")
    pm.add_argument("--out", default=None)

    args = p.parse_args()

    if args.cmd == "eval":
        evaluate_experiment(
            data_dir=args.data_dir, csv_path=args.csv_path,
            checkpoint=args.checkpoint, mode=args.mode,
            image_size=args.image_size, batch_size=args.batch_size,
            key=args.key, block=args.block, out_path=args.out,
            val_seed=args.val_seed,
        )
    elif args.cmd == "mcnemar":
        compare_mcnemar(args.preds_a, args.preds_b, out_path=args.out)
