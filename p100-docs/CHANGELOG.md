# Changelog

Measurements are on 2x Tesla P100-PCIE-16GB (`sm_60`, 180 W cap, driver 575.57.08, CUDA 12.9), model
`Qwen3.8-27B-UD-Q6_K_XL`, `q4_0` KV unless the entry says otherwise, flags
`-ngl 99 -sm tensor -fa 1 -b 2048 -ub 2048 -np 1` with MTP on
(`--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 -ngld 99 -ubd 64`).
[`CHANGES.md`](CHANGES.md) groups the same history by subsystem; this one logs what changed and
what it measured, in time order. Work before 2026-11-19 lives there and in
[`../OPTLOG.md`](../OPTLOG.md).

## 2026-11-23 - a q8_0 cache read directly in the tile kernel

- `4c17c0ec3` adds `flash_attn_tile_load_tile_q8_0`, returns kind 3 from `ggml_cuda_fattn_tile_quant_direct`
  for a same-type q8_0 pair, instantiates `<Q8_0, Q8_0>` in the GQA6 arm, and adds 24 `FLASH_ATTN_EXT` cases.
  Merged as PR 3 (`187cc762e`).

`q8_0` KV with MTP on, ctx 114688 - the conversion arm does not fit a deeper request alongside its own staging.
Same binary, `GGML_CUDA_FA_TILE_Q8_0` off and on, draft acceptance identical at every depth: decode per pass
2k -0.2%, 16k -1.6%, 32k -3.5%, 64k -5.6%, 96k -6.6%, slope 39.4 -> 32.8 ms per 100k tokens, prefill within
1.8%. Free memory over one identical 16k request: 706 MiB per card direct against 262 converting, so 444 MiB
of staging declined - the `sched_reserve` line agrees, 522.00 MiB per card converting at ctx 131072 against
15.25 direct, a slope of 4.07 KiB per token per card. The profile that staging was blocking, a q5_1 target
with a q8_0 draft at ctx 212992, has never loaded on this machine (`graph_reserve` failed an 842 MiB
`cudaMalloc`); with the flag on it loads with 192 MiB free per card and prefills 16001 tokens.

Correctness: bit-exact against the conversion arm, 256/256 identical sampled tokens at shallow depth and 32/32
identical positions over a 101.5k-token chunk, KL 0.000000. `test-backend-ops -o FLASH_ATTN_EXT
-p 'type_K=q8_0'` on both cards: 934 OK / 0 FAIL, 56 of them on the direct loader at the GQA6 shape and 4
per-head-mask cases that must decline it. PTX census for the GQA6 tile kernel: 3664 instructions against 4186
for q5_1, -12.5%, because a q8_0 value is one byte and there is no bit plane to assemble.

Scope, measured: the declined staging is `2 x ctx x D x 2 B` whatever the cache type, so this is worth what
conv tile is worth, not a flat saving. In a traced target-only request 208 of 220 attention ops go to GEMM,
which already reads the quantized cache natively, and there both arms hold 1002 MiB per card after load and 480
minimum across a 108641-token request. As the target cache it also does not move the ceiling: the paired ctx
scan is identical at every rung, deep-prefill wall 155648 with vision off.

## 2026-11-22 - a q5_1 cache read directly in the tile kernel

- `cf3158290` adds `flash_attn_tile_load_tile_q5_1`, turns `ggml_cuda_fattn_tile_q4_0_direct` into
  `ggml_cuda_fattn_tile_quant_direct` returning 0/1/2 per cache type, picks the matching template pair in the
  GQA6 arm, and adds 24 `FLASH_ATTN_EXT` cases. Merged as PR 2 (`409ee5072`).

The tile launcher still passed `need_f16_K/V = true` for q5_1, so every attention call converted the whole
cache and the kernel dequantized from the copy. ctx 163840, MTP on, same binary with
`GGML_CUDA_FA_TILE_Q5_1` off and on: decode per pass 2k -1.0%, 16k -6.6%, 32k -11.5%, 64k -18.5%, 128k
-26.2%, 160k -28.2%, slope 62.16 -> 34.4 ms per 100k tokens with intercepts within 1.3 ms, prefill within
1.1%. Free memory rises 636 MiB per card, about +51k tokens of context ceiling.

Correctness: bit-exact against the conversion arm, 256/256 identical sampled tokens at shallow depth and 64/64
identical positions at 128801 and 87793 tokens, KL 0.000000, with the q4_0 flag pair run as a control on the
pre-existing path. Full CUDA suite: q5_1 4918 OK / 0 FAIL, q4_0 5000 OK / 0 FAIL. Reading the 24 B block as
three 8 B units instead of through `blk->qs` takes the byte loads from 96 to 0 and the kernel from 4717 PTX
ops to 4199, with the fp32 fma count unchanged.

Routing decision that came out of it: the tile-over-vec preference stays scoped to q4_0. Letting it cover q5_1
at `nb <= 2` moves those ops from fp32 products to fp16 ones, which changed 3% of greedy tokens - the vec
kernel already reads q5_1 natively, so tile there buys no work and only changes rounding.

## 2026-11-20 - two attention correctness fixes and the tests that found them

- `880dd9233` gate the fattn vec GQA fold on a broadcast mask. The fold puts several Q heads of one group
  in a block that reads a single mask row through head0, so with a mask that carries head rows every folded
  head reads the same row. The gate already refused to fold when ALiBi gave heads different slopes; it never
  checked the mask. Twelve `FLASH_ATTN_EXT` cases with `mask_heads=2` computed at 0.29..0.81 against a 5e-4
  tolerance before the fix, clean after it.
- `47f1b7f44` make the direct q4_0 tile path agree with the arm that actually runs. The direct predicate
  gates both the vec/tile choice and the whole-cache f16 staging, and was written to mirror `use_gqa_opt`,
  but never checked `mask->ne[2]` - a term upstream added to `use_gqa_opt` inside this merge window, which
  stale-mirrored without a conflict. With head-row masks the fold is skipped, an f16 kernel is launched, and
  it reads the q4_0 buffer reinterpreted as half, past the end of the cache. Probing that family aborts on
  an unpatched tree.
- `f6bd7598b` tests that cross a per-head mask with a quantised KV cache. These are the cases that surfaced
  the two fixes above.

Gate on the merge tree: `test-backend-ops` 4530 OK, 0 FAIL, 0 sentinel, 1345 declined, the same profile as
the pre-merge tree, including the 3 `type_K=q4_0` declines that match the archived baseline. q4_0 kernel
regression sweep over 68 shapes, median -0.02%, no shape at or beyond 1%, largest move 0.88%. End to end at
2k-160k depth with MTP on: decode within 1.6%, prefill within 6.2%.

## 2026-11-20 - upstream sync to `41d8297fe`

`b8ee09f57` merges upstream `ggml-org/llama.cpp` master at `41d8297fe`, 362 commits covering
2026-11-06 to 2026-11-19. Against `p100-optimizations` (`bbad6fcfb`, which is also the merge base) the
whole range is 366 commits, 2248 files, +168468 / -65580. It landed on `p100-optimizations` as PR 1
(`13e3c86a5`, merged 2026-11-22).

Corrects two stale statements in these docs:

- `CHANGES.md` states a 0.3 acceptance threshold from 48k context. The depth rule in
  `common/speculative.cpp` ramps the threshold from 16384 to 49152 positions and tops out at 0.22.
- `QUICKSTART.md` advises `--spec-draft-n-max 3` past about 150k context. Depth is already handled by that
  rule; keep 4, because the q4p decode path refuses verify widths above 5 rows and `n_max 4` lands exactly
  at 5.

## Before this file

Everything up to 2026-10-07 is recorded by subsystem in [`CHANGES.md`](CHANGES.md) - the decode matvec,
the tile and GEMM attention paths, tensor parallelism, cuBLAS precision, CUDA graphs and speculative
decoding, tests, and the earlier upstream merges - with the attempt-by-attempt record, including reverted
work, in [`../OPTLOG.md`](../OPTLOG.md).
