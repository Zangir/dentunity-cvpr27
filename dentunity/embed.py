"""Link embeddings for copy-lineage detection (same recipe as the cached intraoral
link embeddings): whole image resized to 224x224, flip-symmetrised DINOv2 ViT-S/14
class token, L2-normalised. Resumable per group; writes data/linkemb_<group>.npz.

  python -m dentunity.embed <group> <list.txt>   (one image path per line)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .common import DATA

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
HUB = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"


def _tensor(p: str) -> torch.Tensor:
    img = Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)
    a = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (a - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)


def load_model(device):
    m = torch.hub.load(str(HUB), "dinov2_vits14", source="local")
    return m.to(device).eval()


@torch.no_grad()
def embed_paths(paths, model, device, batch=64, out_partial=None):
    out, done = [], 0
    if out_partial and Path(out_partial).exists():
        z = np.load(out_partial)
        out, done = [z["E"]], len(z["E"])
    for s in range(done, len(paths), batch):
        x = torch.stack([_tensor(p) for p in paths[s:s + batch]]).to(device)
        e = model(x) + model(torch.flip(x, [3]))
        out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy().astype(np.float32))
        if out_partial and (s // batch) % 20 == 0:
            np.savez(out_partial, E=np.concatenate(out))
    return np.concatenate(out) if out else np.zeros((0, 384), np.float32)


def main():
    group, lst = sys.argv[1], sys.argv[2]
    paths = [l.strip() for l in open(lst) if l.strip()]
    dst = DATA / f"linkemb_{group}.npz"
    if dst.exists():
        print("exists", dst)
        return
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model(device)
    part = DATA / f"linkemb_{group}.partial.npz"
    E = embed_paths(paths, model, device, out_partial=str(part))
    np.savez(dst, E=E, paths=np.array(paths))
    if part.exists():
        os.remove(part)
    print(group, E.shape)


if __name__ == "__main__":
    main()
