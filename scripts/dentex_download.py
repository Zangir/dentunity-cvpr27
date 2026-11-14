"""Fetch only the DENTEX files we need (fully annotated train subset, validation, test)
from the official Hugging Face page via HTTP range reads. Network only; resumable.
Writes data/dentex/raw/{train,val,test}/ images + label json."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from httpzip import open_zip

BASE = "https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/main/DENTEX/"
OUT = Path(__file__).resolve().parents[1] / "data" / "dentex" / "raw"


def pull(zipname, prefix, dst, want=lambda n: True):
    zf = open_zip(BASE + zipname)
    names = [n for n in zf.namelist() if n.startswith(prefix) and not n.endswith("/") and want(n)]
    dst.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        f = dst / Path(n).name
        if f.exists() and f.stat().st_size == zf.getinfo(n).file_size:
            continue
        f.write_bytes(zf.read(n))
        if i % 50 == 0:
            print(zipname, i, len(names), flush=True)
    return len(names)


if __name__ == "__main__":
    n1 = pull("training_data.zip", "training_data/quadrant-enumeration-disease/", OUT / "train")
    n2 = pull("validation_data.zip", "validation_data/quadrant_enumeration_disease/", OUT / "val")
    pull("validation_data.zip", "validation_data/", OUT / "val", lambda n: n.endswith(".json"))
    import urllib.request, ssl, certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
    vt = OUT / "val" / "validation_triple.json"
    if not vt.exists():
        vt.write_bytes(urllib.request.urlopen(BASE + "validation_triple.json", context=ctx).read())
    n3 = pull("test_data.zip", "disease/", OUT / "test")
    print("done", n1, n2, n3)
