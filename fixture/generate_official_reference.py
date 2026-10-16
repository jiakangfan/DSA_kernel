import numpy as np
from pathlib import Path

SEED = 20260728
CONTEXT = 4096
INDEX_HEADS = 64
INDEX_DIM = 128
TOPK = 2048
ATTN_HEADS = 128
LATENT_DIM = 512
ROPE_DIM = 64
VALUE_DIM = 128

rng = np.random.default_rng(SEED)

# Signed INT4 indexer inputs.
index_q = rng.integers(
    -8, 8, size=(INDEX_HEADS, INDEX_DIM), dtype=np.int8
)
index_k = rng.integers(
    -8, 8, size=(CONTEXT, INDEX_DIM), dtype=np.int8
)

# Per-index-head weights produced by the model's weights_proj path.
index_weights = rng.integers(
    -8, 8, size=(INDEX_HEADS,), dtype=np.int8
)

# Official index score:
# sum_h weight[h] * ReLU(q[h] dot k[token]).
head_dots = (
    index_q.astype(np.int32)
    @ index_k.astype(np.int32).T
)
head_scores = np.maximum(head_dots, 0)
index_scores = (
    index_weights.astype(np.int64)
    @ head_scores.astype(np.int64)
)

# Exact descending score; lower token ID wins a tie.
token_ids = np.arange(CONTEXT, dtype=np.int64)
ranking = np.lexsort((token_ids, -index_scores))
selected = ranking[:TOPK].astype(np.int32)

# Absorbed MLA decode inputs.
q_latent = rng.integers(
    -128, 128, size=(ATTN_HEADS, LATENT_DIM), dtype=np.int16
).astype(np.int8)
q_rope = rng.integers(
    -128, 128, size=(ATTN_HEADS, ROPE_DIM), dtype=np.int16
).astype(np.int8)

latent_cache = rng.integers(
    -128, 128, size=(CONTEXT, LATENT_DIM), dtype=np.int16
).astype(np.int8)
pe_cache = rng.integers(
    -128, 128, size=(CONTEXT, ROPE_DIM), dtype=np.int16
).astype(np.int8)

# Per-head value projection from the shared 512-D latent representation
# to the final 128-D value representation.
value_projection = rng.integers(
    -8, 8,
    size=(ATTN_HEADS, VALUE_DIM, LATENT_DIM),
    dtype=np.int8,
)

selected_latent = latent_cache[selected].astype(np.int64)
selected_pe = pe_cache[selected].astype(np.int64)

scores = (
    q_latent.astype(np.int64) @ selected_latent.T
    + q_rope.astype(np.int64) @ selected_pe.T
)

# High-precision softmax reference for the first functional checkpoint.
scaled = scores.astype(np.float64) / np.sqrt(LATENT_DIM + ROPE_DIM)
scaled -= scaled.max(axis=1, keepdims=True)
probabilities = np.exp(scaled)
probabilities /= probabilities.sum(axis=1, keepdims=True)

weighted_latent = probabilities @ selected_latent.astype(np.float64)

output = np.einsum(
    "hc,hvc->hv",
    weighted_latent,
    value_projection.astype(np.float64),
)

out = Path("official_fixture_ctx4096.npz")
np.savez_compressed(
    out,
    index_q=index_q,
    index_k=index_k,
    index_weights=index_weights,
    index_scores=index_scores,
    selected=selected,
    q_latent=q_latent,
    q_rope=q_rope,
    latent_cache=latent_cache,
    pe_cache=pe_cache,
    value_projection=value_projection,
    attention_scores=scores,
    probabilities=probabilities,
    weighted_latent=weighted_latent,
    output=output,
)

print("REFERENCE PASS")
print("context:", CONTEXT)
print("index heads:", INDEX_HEADS)
print("index head dimension:", INDEX_DIM)
print("top-k:", TOPK)
print("attention heads:", ATTN_HEADS)
print("effective Q/K dimension:", LATENT_DIM + ROPE_DIM)
print("selected first 10:", selected[:10].tolist())
print("selected unique:", np.unique(selected).size)
print("output shape:", output.shape)
print("output checksum:", float(output.sum()))
print("fixture:", out, out.stat().st_size, "bytes")
