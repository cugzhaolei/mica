#!/usr/bin/env python3
"""Verify native_experts.txt offsets against the GGUFs: for every layer x role,
compare the first 4 KiB at the table's absolute offset with the tensor's own
(data_start + tensor offset). Run with a numpy-capable interpreter."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from ds4_pack import load_shards
from gguf_reader import GGUFFile

gguf_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
    "/mnt/models/DeepSeek-V4-Flash-GGUF/UD-IQ2_XXS")
pack = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "/mnt/models/ds4-pack-iq2xxs")

first = sorted(gguf_dir.glob("*-00001-of-*.gguf"))[0]
g = load_shards(first)
T = {t.name: t for t in g.tensors}
paths = sorted(gguf_dir.glob("*.gguf"))
shards = {p.name: np.memmap(p, dtype=np.uint8, mode="r") for p in paths}
ds = {p.name: GGUFFile(p).data_start for p in paths}

ok = bad = 0
for line in (pack / "native_experts.txt").read_text().splitlines():
    if not line.strip() or line.startswith("#"):
        continue
    f = line.split()
    l = int(f[0])
    go, uo, do = (int(x) for x in f[5:8])
    col = f[8].split(",") if len(f) > 8 else []
    if len(col) == 1:
        col = col * 3                      # one shard name for all roles
    for i, (role, want_off) in enumerate(zip(("gate", "up", "down"), (go, uo, do))):
        name = f"blk.{l}.ffn_{role}_exps.weight"
        t = T[name]
        shard = col[i] if i < len(col) else first.name
        mm = shards[shard]
        got = bytes(mm[want_off: want_off + 4096])
        want = bytes(mm[ds[shard] + t.offset: ds[shard] + t.offset + 4096])
        if got == want:
            ok += 1
        else:
            bad += 1
            if bad < 4:
                print("MISMATCH", name, shard, "table_off", want_off, "tensor_off", t.offset, "data_start", ds[shard])
print(f"offset check: {ok} role-layers OK, {bad} bad  (expect 43x3 = 129)")
sys.exit(0 if bad == 0 and ok == 129 else 1)
