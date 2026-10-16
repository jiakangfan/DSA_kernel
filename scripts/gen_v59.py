#!/usr/bin/env python3
"""v59 = v58 + the DEPENDENCE pragma the forwarding registers needed.

v58 added accumulator-forwarding registers to both histogram loops but the II
stayed at 2.  The csynth log gives the reason directly:

    Unable to enforce a carried dependence constraint (II = 1, distance = 1,
    offset = 1) between 'store' ... of variable 'cur_m.V' on array
    'lane_hist_V' and 'load' ... on array 'lane_hist_V'.

The ternary that selects between the forwarded register and the memory read
still issues the memory read unconditionally, so HLS still sees a
read-modify-write to the same runtime address on consecutive iterations and
serialises it.  The registers make the *value* correct; this pragma tells the
scheduler the inter-iteration RAW dependence no longer has to be honoured.

Safe only because v58's bypass is present: at distance 1 the register supplies
the value, and at distance 2 or more the LUTRAM write has already committed.
"""
import sys

SRC = "official_dsa_decode_v58.cpp"
DST = "official_dsa_decode_v59.cpp"

ANCHOR = """#pragma HLS BIND_STORAGE variable=lane_hist type=RAM_2P impl=LUTRAM"""

NEW = ANCHOR + """
// The forwarding registers below supply the value whenever two consecutive
// rows hit the same bin, so the read-modify-write no longer has to be
// serialised against itself.
#pragma HLS DEPENDENCE variable=lane_hist type=inter direction=RAW dependent=false"""

with open(SRC) as f:
    text = f.read()

n = text.count(ANCHOR)
if n != 1:
    sys.exit(f"FAIL: expected 1 BIND_STORAGE anchor, found {n}")
text = text.replace(ANCHOR, NEW)

# the forwarding registers must still be there, or the pragma is unsafe
for need, label in (("fwd_val_m[lane][half]", "forwarding read (mid)"),
                    ("fwd_val_l[lane][half]", "forwarding read (low)"),
                    ("fwd_idx_m[lane][half] = mid;", "forwarding update (mid)"),
                    ("fwd_idx_l[lane][half] = low;", "forwarding update (low)")):
    if need not in text:
        sys.exit(f"FAIL: {label} missing -- pragma would be unsafe")

with open(DST, "w") as f:
    f.write(text)

out = open(DST).read()
assert out.count("HLS DEPENDENCE variable=lane_hist") == 1
print(f"wrote {DST}  ({len(out)} bytes, v58 was {len(open(SRC).read())})")
print("  ok: single DEPENDENCE pragma on lane_hist")
print("  ok: both forwarding bypasses still present")
