# _shared/encrypt_ops.py
import numpy as np
from PIL import Image
import hashlib

# ===== Chaotic XOR 준비 =====
def key_to_seed(key: str) -> float:
    h = hashlib.sha256(key.encode()).hexdigest()
    return (int(h[:8], 16) % 1000) / 1000.0  # 0 ~ 0.999

def logistic_map(length, seed=0.5, r=3.99):
    seq = [seed]
    for _ in range(length - 1):
        seq.append(r * seq[-1] * (1 - seq[-1]))
    return np.array(seq)

def chaotic_xor_encrypt(arr: np.ndarray, key: str) -> np.ndarray:
    flat = arr.flatten()
    seed = key_to_seed(key)
    chaotic = (logistic_map(len(flat), seed=seed) * 255).astype(np.uint8)
    out = np.bitwise_xor(flat, chaotic)
    return out.reshape(arr.shape)

# ===== 픽셀 단위 완전 랜덤 셔플 (per-image) =====
def shuffle_pixels(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[..., None]  # (H,W) -> (H,W,1)
    H, W, C = arr.shape
    perm = rng.permutation(H * W)
    flat = arr.reshape(-1, C)
    shuf = flat[perm].reshape(H, W, C)
    return shuf if C > 1 else shuf[..., 0]

# ===== 블록(패치) 단위 셔플 =====
def shuffle_blocks(arr: np.ndarray, rng: np.random.Generator, block: int = 4) -> np.ndarray:
    """block x block 패치 단위로 섞는다. (H, W)는 block의 배수라고 가정(Resize로 맞추자)"""
    if arr.ndim == 2:
        arr = arr[..., None]
    H, W, C = arr.shape
    assert H % block == 0 and W % block == 0, "H,W must be multiples of block"
    h2, w2 = H // block, W // block
    # (h2, w2, block, block, C)
    tiles = arr.reshape(h2, block, w2, block, C).transpose(0, 2, 1, 3, 4)
    idx = rng.permutation(h2 * w2)
    tiles = tiles.reshape(-1, block, block, C)[idx].reshape(h2, w2, block, block, C)
    # 복원
    out = tiles.transpose(0, 2, 1, 3, 4).reshape(H, W, C)
    return out if C > 1 else out[..., 0]

# ===== 조합 래퍼 =====
def apply_identity(pil_img: Image.Image, **kwargs) -> Image.Image:
    return pil_img

def apply_xor_only(pil_img: Image.Image, key: str, **kwargs) -> Image.Image:
    arr = np.array(pil_img)
    out = chaotic_xor_encrypt(arr, key)
    return Image.fromarray(out)

def apply_shuffle_only(pil_img: Image.Image, img_id=None, **kwargs) -> Image.Image:
    rng = np.random.default_rng(hash(img_id) & 0xFFFFFFFF if img_id is not None else None)
    arr = np.array(pil_img)
    out = shuffle_pixels(arr, rng)
    return Image.fromarray(out)

def apply_partshuffle_only(pil_img: Image.Image, img_id=None, block: int = 4, **kwargs) -> Image.Image:
    rng = np.random.default_rng(hash(img_id) & 0xFFFFFFFF if img_id is not None else None)
    arr = np.array(pil_img)
    out = shuffle_blocks(arr, rng, block=block)
    return Image.fromarray(out)

def apply_xor_then_shuffle(pil_img: Image.Image, key: str, img_id=None, **kwargs) -> Image.Image:
    rng = np.random.default_rng(hash(img_id) & 0xFFFFFFFF if img_id is not None else None)
    arr = np.array(pil_img)
    arr = chaotic_xor_encrypt(arr, key)
    arr = shuffle_pixels(arr, rng)
    return Image.fromarray(arr)

def apply_xor_then_partshuffle(pil_img: Image.Image, key: str, img_id=None, block: int = 4, **kwargs) -> Image.Image:
    rng = np.random.default_rng(hash(img_id) & 0xFFFFFFFF if img_id is not None else None)
    arr = np.array(pil_img)
    arr = chaotic_xor_encrypt(arr, key)
    arr = shuffle_blocks(arr, rng, block=block)
    return Image.fromarray(arr)
