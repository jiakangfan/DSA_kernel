from pathlib import Path
import numpy as np

src = Path("official_fixture_ctx4096.npz")
dst = Path("official_fixture_ctx4096_bin")
dst.mkdir(exist_ok=True)

f = np.load(src)

exports = {
    "index_q.bin":          ("index_q", np.int8),
    "index_k.bin":          ("index_k", np.int8),
    "index_weights.bin":    ("index_weights", np.int8),
    "q_latent.bin":         ("q_latent", np.int8),
    "q_rope.bin":           ("q_rope", np.int8),
    "latent_cache.bin":     ("latent_cache", np.int8),
    "pe_cache.bin":         ("pe_cache", np.int8),
    "value_projection.bin": ("value_projection", np.int8),
    "selected.bin":         ("selected", np.int32),
    "output.bin":           ("output", np.float32),
}

for filename, (key, dtype) in exports.items():
    array = np.ascontiguousarray(f[key], dtype=dtype)
    path = dst / filename
    array.tofile(path)
    print(
        f"{filename:24s}",
        f"shape={str(array.shape):18s}",
        f"dtype={str(array.dtype):8s}",
        f"bytes={path.stat().st_size}",
    )

print("EXPORT PASS:", dst.resolve())
