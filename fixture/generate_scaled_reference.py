"""Float and quantized references sharing one realistic quantization scale.

The raw INT8 dot products span +/-1.1M, so feeding them to exp() directly
saturates the softmax into a pure argmax and leaves the requantization
path untested.  Real quantized inference folds the input scales in:
    softmax_arg = score_int * s_q * s_k / sqrt(576) = score_int / 65536
which the hardware realises as a shift by 11 plus a 256-entry table.
"""

import numpy as np
from pathlib import Path

SEED = 20260728
CONTEXT = 4096
INDEX_HEADS, INDEX_DIM, TOPK = 64, 128, 2048
ATTN_HEADS, LATENT_DIM, ROPE_DIM, VALUE_DIM = 16, 512, 64, 128

LOG2_STEP = 11                  # (max - score) >> 11 indexes the table
LUT_DIVISOR = 32.0              # EXP_LUT[d] = 255 * exp(-d / 32)
SOFTMAX_DENOM = float(2 ** LOG2_STEP) * LUT_DIVISOR   # 65536.0

EXP_LUT = np.array(
    [int(round(255.0 * np.exp(-d / LUT_DIVISOR))) for d in range(256)],
    dtype=np.uint8,
)

rng = np.random.default_rng(SEED)
index_q = rng.integers(-8, 8, (INDEX_HEADS, INDEX_DIM), dtype=np.int8)
index_k = rng.integers(-8, 8, (CONTEXT, INDEX_DIM), dtype=np.int8)
index_weights = rng.integers(-8, 8, (INDEX_HEADS,), dtype=np.int8)

head_dots = index_q.astype(np.int32) @ index_k.astype(np.int32).T
index_scores = (index_weights.astype(np.int64)
                @ np.maximum(head_dots, 0).astype(np.int64))
token_ids = np.arange(CONTEXT, dtype=np.int64)
selected = np.lexsort((token_ids, -index_scores))[:TOPK].astype(np.int32)

q_latent = rng.integers(-128, 128, (ATTN_HEADS, LATENT_DIM),
                        dtype=np.int16).astype(np.int8)
q_rope = rng.integers(-128, 128, (ATTN_HEADS, ROPE_DIM),
                      dtype=np.int16).astype(np.int8)
latent_cache = rng.integers(-128, 128, (CONTEXT, LATENT_DIM),
                            dtype=np.int16).astype(np.int8)
pe_cache = rng.integers(-128, 128, (CONTEXT, ROPE_DIM),
                        dtype=np.int16).astype(np.int8)
value_projection = rng.integers(-8, 8,
                                (ATTN_HEADS, VALUE_DIM, LATENT_DIM),
                                dtype=np.int8)

selected_latent = latent_cache[selected].astype(np.int32)
selected_pe = pe_cache[selected].astype(np.int32)

scores = (q_latent.astype(np.int32) @ selected_latent.T
          + q_rope.astype(np.int32) @ selected_pe.T)

# ---------- float path, same scale ----------
arg = scores.astype(np.float64) / SOFTMAX_DENOM
arg -= arg.max(axis=1, keepdims=True)
prob_f = np.exp(arg)
prob_f /= prob_f.sum(axis=1, keepdims=True)
weighted_f = prob_f @ selected_latent.astype(np.float64)
output_f = np.einsum("hc,hvc->hv", weighted_f,
                     value_projection.astype(np.float64))

# ---------- quantized path ----------
distance = np.minimum(
    (scores.max(axis=1, keepdims=True) - scores) >> LOG2_STEP, 255)
prob_u8 = EXP_LUT[distance.astype(np.int64)]
denominator = prob_u8.astype(np.int64).sum(axis=1)

weighted_int = prob_u8.astype(np.int64) @ selected_latent.astype(np.int64)
magnitude = np.abs(weighted_int).max(axis=1)
shift = np.array([max(0, int(m).bit_length() - 7) for m in magnitude],
                 dtype=np.int32)
weighted_i8 = np.empty((ATTN_HEADS, LATENT_DIM), dtype=np.int8)
for head in range(ATTN_HEADS):
    weighted_i8[head] = np.clip(
        weighted_int[head] >> shift[head], -128, 127).astype(np.int8)

output_int = np.einsum("hc,hvc->hv", weighted_i8.astype(np.int32),
                       value_projection.astype(np.int32))
dequant = (2.0 ** shift) / denominator
output_q = output_int.astype(np.float64) * dequant[:, None]

cos = float((output_f.ravel() @ output_q.ravel())
            / (np.linalg.norm(output_f) * np.linalg.norm(output_q)))
rel = float(np.abs(output_f - output_q).max() / np.abs(output_f).max())

out = Path("official_fixture_ctx4096_scaled.npz")
np.savez_compressed(
    out, exp_lut=EXP_LUT, log2_step=LOG2_STEP,
    softmax_denom=SOFTMAX_DENOM, selected=selected,
    attention_scores=scores.astype(np.int32),
    probabilities_u8=prob_u8, denominator=denominator.astype(np.int64),
    weighted_int=weighted_int.astype(np.int32), shift=shift,
    weighted_i8=weighted_i8, output_int=output_int.astype(np.int32),
    output_float=output_f, output_quant=output_q,
)
binary_dir = Path("official_fixture_ctx4096_bin")
binary_dir.mkdir(exist_ok=True)
output_q.astype(np.float64).tofile(binary_dir / "output_scaled_quant.bin")
output_f.astype(np.float64).tofile(binary_dir / "output_scaled_float.bin")

print("SCALED REFERENCE PASS")
print("softmax denominator      :", SOFTMAX_DENOM)
print("nonzero LUT entries      :", int((EXP_LUT > 0).sum()))
print("softmax arg std          : %.3f" % float(arg.std()))
print("distance range           :", int(distance.min()), int(distance.max()))
print("nonzero probs per head   :", int((prob_u8 > 0).sum(axis=1).mean()))
print("denominator range        :", int(denominator.min()),
      int(denominator.max()))
print("weighted_int abs max     :", int(np.abs(weighted_int).max()))
print("shift per head           :", shift.tolist())
print("output_int abs max       :", int(np.abs(output_int).max()))
print()
print("--- quantization loss vs float, same scale ---")
print("cosine similarity        : %.10f" % cos)
print("max relative error       : %.6e" % rel)
print("RMSE                     : %.6e"
      % float(np.sqrt(((output_f - output_q) ** 2).mean())))
