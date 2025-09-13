# _shared/gradcam.py
import os, sys, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

# 내부 모듈 경로
sys.path.append(os.path.dirname(__file__))
from train_util import default_transform, build_model
from dataset import RSNADataset
from encrypt_ops import (
    apply_identity, apply_shuffle_only, apply_partshuffle_only,
    apply_xor_only, apply_xor_then_shuffle, apply_xor_then_partshuffle
)

def get_preprocess_fn(mode):
    mode = mode.lower()
    if mode in ["basic"]:
        return apply_identity, {}
    if mode in ["pixelshuffle-only"]:
        return apply_shuffle_only, {}
    if mode in ["partshuffle-only"]:
        return apply_partshuffle_only, {}
    if mode in ["chaotic-only"]:
        return apply_xor_only, {}
    if mode in ["chaotic_pixelshuffle"]:
        return apply_xor_then_shuffle, {}
    if mode in ["chaotic_partshuffle"]:
        return apply_xor_then_partshuffle, {}
    raise ValueError(mode)

@torch.no_grad()
def _forward_features(model, x):
    # 필요시 확장할 수 있는 헬퍼
    return model(x)

def gradcam_on_batch(model, x, target_layer):
    """
    Grad-CAM 계산 (B개 샘플)
    반환: cams [B, Hc, Wc] (원본 feature map 해상도), preds [B]
    """
    model.eval()
    features = {}
    grads = {}

    def f_hook(_, __, out): features["value"] = out.detach()
    def b_hook(_, grad_in, grad_out): grads["value"] = grad_out[0].detach()

    # 후킹
    h1 = target_layer.register_forward_hook(f_hook)
    h2 = target_layer.register_full_backward_hook(b_hook)

    with torch.enable_grad():
        out = model(x)
        cls = out.argmax(1)
        score = out.gather(1, cls.view(-1, 1)).sum()
        model.zero_grad()
        score.backward()

    fmap = features["value"]               # [B, C, Hc, Wc]
    grad = grads["value"]                  # [B, C, Hc, Wc]
    weights = grad.mean(dim=(2, 3), keepdim=True)   # [B, C, 1, 1]
    cam = (weights * fmap).sum(dim=1, keepdim=True) # [B, 1, Hc, Wc]
    cam = torch.relu(cam)

    cams = []
    B = cam.size(0)
    for i in range(B):
        c = cam[i, 0].detach().cpu().numpy()   # [Hc, Wc]
        c = (c - c.min()) / (c.max() - c.min() + 1e-8)
        cams.append(c)

    # 후크 해제
    h1.remove(); h2.remove()
    return np.stack(cams, axis=0), cls.detach().cpu().numpy()

def save_gradcam_grid(imgs, cams, preds, gts, out_path, ncols=4):
    """
    imgs: torch.Tensor [B, 3, H, W]  (정규화된 입력)
    cams: np.ndarray  [B, Hc, Wc]    (Grad-CAM 원 해상도)
    """
    # 입력 복원
    x = imgs.detach().cpu().numpy()
    x = (x * 0.5 + 0.5)  # de-normalize from [-1,1] to [0,1]
    x = np.clip(x, 0, 1)

    B, _, H, W = x.shape
    nrows = int(np.ceil(B / ncols))
    fig = plt.figure(figsize=(ncols * 3, nrows * 3))

    for i in range(B):
        img = (x[i].transpose(1, 2, 0) * 255).astype(np.uint8)  # [H, W, 3]

        # ★ 핵심 수정: CAM(예: 7x7)을 입력 이미지 해상도(예: 224x224)로 업샘플
        cam_resized = cv2.resize(cams[i], (W, H), interpolation=cv2.INTER_CUBIC)  # [H, W]
        heat = (cam_resized * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

        # Overlay
        over = (0.4 * heat + 0.6 * img).astype(np.uint8)

        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.imshow(over)
        ax.set_title(f"pred={preds[i]} / gt={gts[i]}")
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def run_gradcam_for_experiment(exp_dir, ckpt_path, mode, data_dir, csv_path,
                               key=None, block=8, image_size=224, batch_size=8,
                               val_seed=42, num_samples=12, out_dir=None):
    """
    단일 실험(폴더) 기준으로 Grad-CAM 샘플 그리드 저장
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    if out_dir is None:
        out_dir = Path(exp_dir) / "gradcam"
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    # 동일 split 유지
    df = pd.read_csv(csv_path)
    label_map = {str(r["patientId"]): int(r["Target"]) for _, r in df.iterrows()}
    ids = sorted(list(label_map.keys()))
    _, val_ids = train_test_split(ids, test_size=0.2, random_state=val_seed)

    tfm = default_transform(image_size)
    preprocess_fn, kwargs = get_preprocess_fn(mode)
    # chaotic 계열 key / part 계열 block 전달
    if "chaotic" in mode and key is not None:
        kwargs["key"] = key
    if "part" in mode:
        kwargs["block"] = block

    ds = RSNADataset(data_dir, label_map, transform=tfm,
                     filter_ids=val_ids, preprocess_fn=preprocess_fn,
                     preprocess_kwargs=kwargs)

    # 임의 샘플 추출
    num = min(num_samples, len(ds))
    idxs = random.sample(range(len(ds)), k=num)
    xs, ys = zip(*[ds[i] for i in idxs])
    x = torch.stack(xs, dim=0)  # [B, 3, H, W]
    y = torch.stack(ys, dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    # ResNet-18 마지막 conv 레이어
    target_layer = model.layer4[-1].conv2

    # CAM 계산 (원 해상도), 이후 save 함수에서 업샘플하여 합성
    cams, preds = gradcam_on_batch(model, x.to(device), target_layer)

    out_path = Path(out_dir) / "gradcam_grid.png"
    save_gradcam_grid(x, cams, preds, y.numpy(), out_path)
    print(f"✅ Saved Grad-CAM -> {out_path}")
