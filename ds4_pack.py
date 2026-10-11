#!/usr/bin/env python3
"""ds4_pack.py — DeepSeek V4 Flash GGUF -> Strata pack (P1 skeleton).

Reuses tools/iq_pack.py's layout verbatim; the only model-specific parts are:
  1. expert tensor detection — llama.cpp DEEPSEEK4 names experts
     blk.N.ffn_{gate,up,down}_exps — SAME suffix as qwen4exp, so iq_pack's
     is_expert() (endswith '_exps.weight') already matches.
  2. native_experts.txt blob layout — qwen packs [gate|up|down] rows per
     expert in one fused tensor; DEEPSEEK4 GGUF has three separate fused
     tensors (gate_exps/up_exps/down_exps, n_expert rows each). This builder
     interleaves them into per-expert blobs [gate rows | up rows | down rows]
     exactly like iq_pack's experts.bin form.
  3. arch keys — engine reads qwen4exp.* today; we emit BOTH qwen4exp.* (for
     the current loader) and deepseek4.* (for the P2 branch) in the pack
     metadata until generate.cpp learns the new prefix.

Status: P1 skeleton — dense.bin/index.txt/conversions.json paths are
iq_pack-compatible; run `--dry-run` against a GGUF to validate tensor census.
"""
import argparse, json, pathlib, sys

# DEEPSEEK4 -> Strata engine layer-role mapping (what generate.cpp must route):
ROLE_MAP = {
    # attention (MLA variant + CSA/HCA compressors + Lightning Indexer)
    "blk.N.attn_q_a":        "MLA q down-proj -> resident (dense.bin)",
    "blk.N.attn_q_b":        "MLA q up-proj   -> resident",
    "blk.N.attn_kv":         "KV down-proj (shared c_kv) -> resident",
    "blk.N.attn_out_a/b":    "output down/up split -> resident",
    "blk.N.attn_compressor_*": "CSA/HCA segment compressor (wkv/wgate/ape/norm) -> resident, NEW kernel P2",
    "blk.N.indexer_*":       "Lightning Indexer (proj + compressor) -> resident, maps to Strata QSA indexer kernels",
    "blk.N.hc_*":            "mHC hyper-connection (fn/base/sinkhorn scale) -> resident, NEW tiny kernel P2",
    # MoE
    "blk.N.ffn_gate_inp":    "router -> resident",
    "blk.N.ffn_gate_up_down_exps": "256 routed experts -> PACK EXPERT POOL (tiered)",
    "blk.N.ffn_*_shexp":     "1 shared expert -> resident (like qwen shared expert)",
    # MTP/DSpark
    "nextn.*":               "DSpark/MTP draft module -> --mtp rt equivalent (P3)",
}

def census(gguf_path, dry=True):
    """Tensor census: count experts vs dense, verify suffix compatibility."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))
    try:
        from gguf_reader import GGUFFile  # vendored in Strata tools/
    except ImportError:
        print("run from /mnt/windows/Strata/tools (needs gguf_reader.py)"); return 1
    g = GGUFFile(str(gguf_path))
    exps = [t for t in g.tensors if t.name.startswith("blk.") and t.name.endswith("_exps.weight")]
    dense = [t for t in g.tensors if t not in exps]
    print(f"tensors: {len(g.tensors)}  expert-fused: {len(exps)}  dense: {len(dense)}")
    for t in exps[:3]:
        print("  expert:", t.name, t.type_name, t.shape)
    need = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}
    seen = {t.name.split(".", 2)[-1] if t.name.count(".") >= 2 else t.name for t in exps}
    missing = need - seen
    if missing:
        print("!! unexpected expert naming, missing:", missing); return 2
    print("suffix check OK: DEEPSEEK4 *_exps.weight matches iq_pack is_expert()")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", nargs="?")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.gguf:
        print(__doc__); sys.exit(0)
    sys.exit(census(a.gguf))
