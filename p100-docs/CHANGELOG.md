# Changelog

Measurements are on 2x Tesla P100-PCIE-16GB (`sm_60`, 180 W cap, driver 575.57.08, CUDA 12.9), model
`Qwen3.8-27B-UD-Q6_K_XL`, `q4_0` KV unless the entry says otherwise, flags
`-ngl 99 -sm tensor -fa 1 -b 2048 -ub 2048 -np 1` with MTP on
(`--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 -ngld 99 -ubd 64`).
[`CHANGES.md`](CHANGES.md) groups the same history by subsystem; this one logs what changed and
what it measured, in time order. Work before 2026-11-19 lives there and in
[`../OPTLOG.md`](../OPTLOG.md).

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
whole range is 366 commits, 2248 files, +168468 / -65580.

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
