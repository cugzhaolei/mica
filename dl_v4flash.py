#!/usr/bin/env python3
"""Download unsloth/DeepSeek-V4-Flash-GGUF UD-IQ2_XXS shards via hf-mirror (resumable)."""
import os, sys, time, urllib.request, json

REPO = "unsloth/DeepSeek-V4-Flash-GGUF"
FOLDER = "UD-IQ2_XXS"
BASE = "https://hf-mirror.com"
OUT = "/mnt/models/DeepSeek-V4-Flash-GGUF/UD-IQ2_XXS"
os.makedirs(OUT, exist_ok=True)

def get(url, hdr=None):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.5", **(hdr or {})})
    return urllib.request.urlopen(req, timeout=60)

files = json.loads(get(f"{BASE}/api/models/{REPO}/tree/main/{FOLDER}").read())
targets = [f["path"] for f in files if f["path"].endswith((".gguf", ".gguf.split", ".txt", ".md"))]
print("files:", len(targets))

for p in targets:
    url = f"{BASE}/{REPO}/resolve/main/{p}"
    dst = os.path.join(OUT, os.path.basename(p))
    have = os.path.getsize(dst) if os.path.exists(dst) else 0
    try:
        total = int(get(url, {"Range": "bytes=0-0"}).headers.get("Content-Range", "*/0").split("/")[1])
    except Exception:
        total = 0
    if have and have == total:
        print(f"skip {os.path.basename(p)} ({have} complete)"); continue
    print(f"get {os.path.basename(p)} from {have}/{total}")
    for attempt in range(10):
        try:
            have = os.path.getsize(dst) if os.path.exists(dst) else 0
            r = get(url, {"Range": f"bytes={have}-"})
            with open(dst, "ab") as fh:
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk: break
                    fh.write(chunk)
            got = os.path.getsize(dst)
            if total and got != total: raise RuntimeError(f"short {got}/{total}")
            print(f"done {os.path.basename(p)} {got}"); break
        except Exception as e:
            print(f"  retry{attempt} {e}"); time.sleep(5 * (attempt + 1))
    else:
        sys.exit(f"FAILED {p}")
print("ALL_DONE")
