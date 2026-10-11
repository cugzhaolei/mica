# deepseek4-strata

Engineering framework to adapt [Strata-style MoE expert tiering](GPU→RAM→SSD offload + speculative decoding)
to **DeepSeek V4 Flash** (284B total / 13B active, 256 experts top-6, 43 layers, 1M ctx)
on **2× RTX 2080Ti 22G (Turing sm_75) + 157G RAM**.

First public **sm_75 benchmark data** for DeepSeek V4 Flash is in [DESIGN.md](DESIGN.md) (P0 section).

## Contents

| File | What |
|---|---|
| `DESIGN.md` | Full design: source recon, arch deltas, 5-phase roadmap, risk ledger, **P0 measured results** |
| `ARTICLE.md` | Chinese write-up (also published on iot-pub.top) |
| `ds4_pack.py` | P1 skeleton: GGUF → Strata expert-pack builder (census verified: DEEPSEEK4 `*_exps.weight` naming matches) |
| `gguf_reader.py` | Minimal GGUF metadata/tensor-index reader (no deps) |
| `dl_v4flash.py` | Resumable hf-mirror downloader for unsloth UD-IQ2_XXS shards |

## P0 baseline (llama.cpp, sm_75, first public numbers)

```
llama-server -m DeepSeek-V4-Flash-UD-IQ2_XXS-00001-of-00003.gguf \
  -ngl 99 -ncmoe 99 -cram 6 -b 2048 -ub 512 -fa 0 -c 8192
```

| metric | baseline | tuned |
|---|---|---|
| prefill | 6.5–8 tok/s | **73 tok/s** (×11) |
| decode | 3.2→4.8 tok/s | ~3 (fa on Turing hurts decode; avoid `-fa 1`) |

Conclusion: llama.cpp static placement caps decode at ~5 tok/s on this hardware
(CPU expert GEMM, no LRU). The Strata ExpertSource tiered cache (P2) targets
**25–45 tok/s decode / 300–800 tok/s prefill** by keeping hot experts in VRAM.

## Status

- [x] P0 llama.cpp baseline + sm_75 measurements
- [ ] P1 expert-pack builder (`ds4_pack.py`)
- [ ] P2 engine arch branch (CSA/HCA compressed attention, mHC-Sinkhorn, QSA indexer reuse)
- [ ] P3 DSpark speculative decoding
- [ ] P4 tuning (GPU1 expert LRU, 1M ctx, conversation cache)

Model: [unsloth/DeepSeek-V4-Flash-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF) (UD-IQ2_XXS, 90.9G).
Write-up (Chinese): https://iot-pub.top/llm/deepseek4-strata.html