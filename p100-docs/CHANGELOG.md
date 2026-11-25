# Changelog

Measurements are on 2x Tesla P100-PCIE-16GB (`sm_60`, 180 W cap, driver 575.57.08, CUDA 12.9), model
`Qwen3.8-27B-UD-Q6_K_XL`, `q4_0` KV unless the entry says otherwise, flags
`-ngl 99 -sm tensor -fa 1 -b 2048 -ub 2048 -np 1` with MTP on
(`--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 -ngld 99 -ubd 64`).
[`CHANGES.md`](CHANGES.md) groups the same history by subsystem; this one logs what changed and
what it measured, in time order. Work before 2026-11-19 lives there and in
[`../OPTLOG.md`](../OPTLOG.md).

## 2026-11-25 - upstream sync to `beb9d4933`

`229df098e` merges upstream `ggml-org/llama.cpp` master at `beb9d4933` (version 1.1.0), 209 commits
since the previous merge point `41d8297fe`. Against `p100-optimizations` (`0a4e3dd04`, also the merge
base) the head is 6 commits, 1093 files, +76347 / -22452. It landed on `p100-optimizations` as PR 7
(`dff1f4225`, merged 2026-11-25).

The hard file was again `gated_delta_net.cu`: upstream rewrote it with pending-token replay
(`ggml_gated_delta_net_ext`, `src[6]`) and a 4-cols-per-warp layout. Resolution keeps the upstream
kernel and re-applies our Pascal features on top (state-gather fusion, fused l2 pre-op, chunked alt
path). `ggml-backend.cpp` hand-combines the fork input-copy sync reduction with upstream's
`fast_async_host_input` async host reads.

Five fork fixes ride with the merge, each caught by a battery leg:

- `1c0ced0a9` our chunked GDN form (`GGML_CUDA_GDN_CHUNKED`, default on) assumed packed contiguous
  q/k/v/state. Upstream's padded shapes made it read at wrong strides and its pending input was
  silently dropped (ERR ~1e-3). Now runs only on contiguous, pending-free inputs.
- `f1d64244c` `--spec-draft-ubatch-size` lacked `set_spec()`; upstream's arg-parser test
  excepts spec args from alias ordering.
- `7468e47b9` the compact I32 causal mask reached the lightning indexer, which reads mask values and
  asserts f16; deepseek4 lightning/DSA aborted at graph build. Compact branch is now opt-out.
- `4123d1877` same class: gemma4 prune copies dense mask rows, minimax-m3 MSA reads and casts the
  mask. `llm_attn_mask.force_f16` requests the dense form; mask reuse honours it.
- `0a09c55e8` fork graphs keep node and leaf counts across ubatches while a wider ubatch grows the
  mask in bytes, so the `GGML_SCHED_DEBUG_REALLOC=1` heuristic (set by ctest) judged that realloc
  unexpected and aborted on clover_moe `noalloc`. The judgement counts bytes next to counts; a truly
  unreserved realloc still aborts. This one is inherited from the Kmic base, which aborts identically;
  stock never fires it because its node counts vary.

Battery on the merge tree, 2x P100: `test-llama-context` 150/150 serial (150 dummy archs),
`test-backend-ops -b CUDA0` 21,811 OK / 0 FAIL, `FLASH_ATTN_EXT` family 4,625 / 0, `GATED_DELTA_NET`
incl. replay 111 / 0, ctest full suite 75/75 rc=0 (torch and git-lfs legs now runnable on this box),
VQ80 dispatch proof 465 OK / 0 FAIL / 0 sentinel at both `GGML_CUDA_FA_Q4P_Q8V` positions, bench
advisory UD-Q6_K_XL q8_0 KV pp512 245.65 +/- 0.41, pp2048 367.23 +/- 0.90. Known non-gate: grad mode
aborts upstream on split-REGLU (`ggml.c:7303`), a stock registration bug unrelated to this fork.

## 2026-11-25 - the q4p kernel running on a q8_0 V cache

- `083bf471b` teaches `flash_attn_ext_q4p` a q8_0 V through the `VQ80` template flag, gates the V side
  independently of K (q8_0 V pairs with q4_0, q5_1 and q8_0 K), and adds q8_0 V eval cases against each
  K path plus a constant fill (`d = 1, qs[j] = j`, so dim j is `j - 128` and every dim, word and
  half-word differs). Runtime switch `GGML_CUDA_FA_Q4P_Q8V`, default on. Merged as PR 5 (`cbd2e77d8`).

Before this, a full q8_0 cache had no in-register path: K learned q8_0 earlier, V refused it and the
whole call fell back - `nb <= 2` to vec, above that to the tile kernel PR 3 made direct.

Kernel A/B at `kv=262144`, `nh=8`, `us/run`, min of two reps (VQ80 on against the direct tile kernel):
nb=1 1160 against 2505, nb=2 1397 against 3193, nb=3 2374 against 3317, nb=4 2446 against 3310,
nb=5 2829 against 4968 - 2.16x, 2.29x, 1.40x, 1.35x, 1.76x. The flag switch itself is proven by timing:
the same shape runs 1157 us with `GGML_CUDA_FA_Q4P_Q8V=1` and 2519 us with `=0`.

End to end with q8_0 KV, draft cache q4_0, ctx 114688, MTP on, per-pass latency (acceptance-free
`ms/pass`, acceptance matched .652/.652 at 96k): 16k +3.5%, 32k +5.4%, 64k +11.1%, 96k +17.3%
(87.25 -> 72.20 ms/pass); flat at 2k. The same arms with q5_1 KV measure +17.2% at 96k, so the two
V-flags behave identically end to end.

Correctness: full `FLASH_ATTN_EXT` eval 4625 OK / 0 FAIL / 0 sentinel, dispatch proof green at both
switch positions; q8_0 V family max NMSE 1.229e-05 against the fp32 reference (tolerance 5e-4).
Determinism: 256/256 replay identical at KL 0.000. Deep MTP replay at measured depth 96699: 32/32
sampled tokens identical to the tile path, mean KL 3.4e-5, max 6.9e-4. Shallow greedy probe: 225/256
identical, 31 flips all near-tied (median non-flipped logprob delta 0.0025 nats).

A q8_0 V block is 34 B, a 2-byte scale plus 32 bytes, so the loader gets its own split-prefetch word
offset (`pv_o8`): the thread's two words, dims `4k..4k+3` and `16+4k..4k+3`, are read 16 B apart. K
geometry is unchanged. VQ80 runs the plain PV path only - the RS shuffles assume nibble and 5th-bit
layouts, which a byte-per-dim block does not have. Two bugs found while validating, both fixed: the
H16PV q8_0 dequant was missing the sign-bit flip (`^ 0x80808080u`), and the prefetch advanced the
block pointer before capturing `wA/wB` into `wvA/wvB`, skewing every block after the first - the
constant fill localizes both.

Open: register spill at `R=18 nb=3` (120 B with q5_1 K, 132 B with q8_0 K) and `R=30 nb=5` (20 B with
q4_0 K) per ptxas. Spill is bit-exact storage and the spilling widths still measure 1.40x and 1.76x
ahead of the tile kernel; the shrink alternative (`R=15`) was already spilling before this change and
is slower. Kept as a tripwire for future register additions.

## 2026-11-24 - the q4p kernel running on a q5_1 cache

- `f9364850d` teaches `flash_attn_ext_q4p` a q5_1 K and V through the `KQ5` / `VQ51` template flags,
  takes the block geometry from `fattn_q4p_cfg<R, OLD, KQ8, KQ5>` (KBB 24, NW 12, NSPLIT 4), gates the
  two cache sides independently, and adds 23 `FLASH_ATTN_EXT` eval cases plus 3 perf shapes, including
  constant-fill ones. `7e6f4302f` fixes a stale `GGML_CUDA_FA_VEC_MAX_Q` comment. Merged as PR 4
  (`b451f03c9`).

Before this, a q5_1 cache had no in-register path: `nb <= 2` went to vec, above that to the tile kernel
that PR 2 made direct. `q4_0` had `q4p`, worth +27% of decode, and it refused q5_1 because the kernel was
built on the 18 B q4_0 block and required a q4_0 V. Runtime switch `GGML_CUDA_FA_Q4P_Q51`, default on.

Kernel A/B at `kv=262144`, `nh=8`, `us/run`, best of two reps (on against off): nb=1 963 against 2565,
nb=2 1343 against 7295, nb=3 1811 against 3071, nb=5 3073 against 5185 - 2.66x, 5.45x, 1.70x, 1.69x.
Against the same q4p shape with q4_0 KV (638 / 943 / 1515 / 1830) the ratio is 1.51 / 1.43 / 1.20 /
1.68, and the byte ratio is 1.33, so nb 1-3 are near bandwidth-bound and nb=5 is not.

End to end with q5_1 KV, draft cache q4_0, ctx 212992, MTP on, greedy, `LLAMA_SPEC_SAMPLE_TEMP=1.0`
(so the draft chain is sampled and acceptance is not the greedy default), per-pass latency (`ms/token x
accepted mean length`, not raw `t/s`): 14499 -3.6%, 65812 -10.8%, 130547 -16.6%, 196605 -18.4%; slope
33.55 -> 21.58 ms per 100k tokens with intercepts 66.7 against 65.9. Without MTP the single-token pass is
34.7 against 39.5 ms at ~1.3k depth and 39.0 against 50.8 ms at 99215. A q4_0 cache on q4p measures the
same 34.7 ms as q5_1 on q4p and q8_0 on vec is equally slow, so that gap is vec's fixed per-call cost
rather than cache bytes - the vec kernel was never cheap at nb=1 regardless of what it reads.

Correctness: full `FLASH_ATTN_EXT` eval 4601 OK / 0 FAIL / 0 NaN; q5_1 family 375 OK, q4_0 428 OK, both
mixed sides 3 OK each. NMSE against the CPU reference 1.9-2.4e-6 for q4p-q5_1, against 2.2-3.1e-6 for
tile-q5_1 and 2.5e-6 for q4_0 q4p, tolerance 5e-4. Greedy logits A/B over a 160-position MTP ladder:
159/160 identical sampled tokens, mean KL 0.001153, non-flip median 1e-6, and the same arm replayed
against itself 160/160 at KL 0.000000. Not bit-exact and cannot be: this is a different kernel, so the
accumulation order differs, and PR 2's own routing note (tile over vec at nb<=2 changed 3% of greedy
tokens) is the same effect. At nb=1 the move is from vec's fp32 PV accumulation to fp16, which cost
34/256 flips on a no-MTP probe at pre-flip median KL 5.3e-5 - kept, because under MTP a nb=1 target pass
needs a zero-length draft chain and production always runs MTP.

Two things the debugging left behind, both worth remembering. The 5th bit of a q5_1 value is indexed by
the dim index inside the block (dim j is `qh` bit j, dim j+16 is `qh` bit j+16, as
`ggml_dequantize_row_q5_1` does); a nibble-sequential index looks equivalent and shifts the value by
exactly `16 * d` on the affected dims, dims `28 mod 32`. And dropping `KQ5` from the cfg alias makes the
kernel read q4_0-sized blocks: KBB 18, NW 9, split stride 36 B, KPF wrongly enabled, and NaN. V is
dequantized with an fp32 `fmaf` and one fp16 round rather than in pure fp16, because `x * d + m` cancels
near zero - `tools/proofs-fattn-q4_0/q51_dequant_proof.cpp` puts that at max relative error 4.88e-4 above
2^-14 and 4 differing samples in 332M.

Context ceilings are unchanged: q5_1 loads and prefills 235520, loads 237568, and 262144 still refuses by
the same ~15 MiB per card at `h_nextn_dev_alloc`, which no attention kernel touches. Verified by loading
both 237568 and 262144 with the switch on, and 262144 with it off, for the identical failure.

Open item: nb=5 sits at 1.68x q4_0 instead of 1.33x. Not register spill - ptxas shows no q5_1
instantiation spilling ((5,3) 168 registers, (4,6) 255, 0 bytes), and the spilling ones are the
pre-existing q4_0 instantiations - and not the `ncols2=3` half-group config, since `GGML_CUDA_Q4P_NC2=6`
measures slower than the default (3528 against 3096 us). The cause is that `KQ5` disables `H16QK` and
`KPF`, so QK products stay fp32; enabling them for q5_1 moves QK to fp16 and needs its own validation pass.

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
