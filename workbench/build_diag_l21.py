"""Build diag_l21_offsets.py from diag_oracle.py (oracle +-24 variant).

Inserts a per-call winning-offset histogram dump for the ONLINE q/k/v
dynamic encoders only (calibration calls carry no tag and are skipped).
Output: logs/execution/diag_l21_offsets.jsonl  (one line per online call).
"""
import io

SRC = r"logs\execution\diag_oracle.py"
DST = r"logs\execution\diag_l21_offsets.py"

s = open(SRC, encoding="utf-8").read()

# 1) module-level tag
anchor1 = "_REFINE_EDGE_EXTENSION = True"
assert anchor1 in s
s = s.replace(
    anchor1,
    anchor1 + "\n_DIAG_API_TAG = None\n_DIAG_DUMP = r\"logs\\execution\\diag_l21_offsets.jsonl\"",
    1,
)

# 2) dump point after edge extension, before L1 block
anchor2 = "        extend_edge(hi_offset, +1)\n        extend_edge(lo_offset, -1)\n"
assert anchor2 in s
dump_code = anchor2 + """
        if _DIAG_API_TAG is not None:
            import json as _dj
            _counts = torch.bincount(
                best_offset.to(torch.int64) + 260, minlength=520
            )
            _hist = {
                str(int(_k) - 260): int(_v)
                for _k, _v in enumerate(_counts.tolist())
                if _v > 0
            }
            with open(_DIAG_DUMP, "a", encoding="utf-8") as _df:
                _df.write(
                    _dj.dumps(
                        {
                            "api": _DIAG_API_TAG,
                            "n_hard": int(best_offset.numel()),
                            "hist": _hist,
                        }
                    )
                    + "\\n"
                )
"""
s = s.replace(anchor2, dump_code, 1)

# 3) tag wrappers on the three online entry points
def wrap(fn_name, api):
    a = f"@torch.no_grad()\ndef {fn_name}(\n"
    assert a in s, fn_name
    return a

for fn, api in (
    ("hif4_dynamic_quantize_q", "q"),
    ("hif4_dynamic_quantize_k", "k"),
    ("hif4_dynamic_quantize_v", "v"),
):
    # find the single `    return _nvfp4_to_hif4(` inside each fn body:
    # split at def, wrap the return.
    head = f"def {fn}(\n"
    i = s.index(head)
    j = s.find("def ", i + 10)
    if j == -1:
        j = len(s)
    body = s[i:j]
    ret = "    return _nvfp4_to_hif4(\n"
    assert ret in body, fn
    new_body = body.replace(
        ret,
        f"""    global _DIAG_API_TAG
    _diag_prev, _DIAG_API_TAG = _DIAG_API_TAG, "{api}"
    _diag_result = _nvfp4_to_hif4(\n""",
        1,
    )
    # the call ends with a line '    )' -- append tag restore + return
    k = new_body.index("_diag_result = _nvfp4_to_hif4(")
    m = new_body.index("\n    )\n", k)
    new_body = (
        new_body[: m + 1]
        + "    )\n"
        + "    _DIAG_API_TAG = _diag_prev\n"
        + "    return _diag_result\n"
        + new_body[m + len("\n    )\n") :]
    )
    s = s[:i] + new_body + s[j:]

io.open(DST, "w", encoding="utf-8", newline="\n").write(s)
print("written", DST, len(s))
