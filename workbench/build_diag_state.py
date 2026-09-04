"""Build state-dump variants: oracle(+-24) vs v186(5-code) calibration states."""
import io
import re

FUNC_CODE = '''def _diag_dump_states(q_state, k_state, v_state):
    import json as _sj

    def _state_summary(_st):
        _out = {}
        for _k, _v in _st.items():
            if isinstance(_v, torch.Tensor):
                if _v.numel() > 16:
                    _out[_k] = {
                        "shape": list(_v.shape),
                        "l2": float(_v.float().norm().item()),
                        "head": [
                            round(float(_x), 6)
                            for _x in _v.float().flatten()[:6].tolist()
                        ],
                    }
                else:
                    _out[_k] = [
                        round(float(_x), 6)
                        for _x in _v.float().flatten().tolist()
                    ]
            elif isinstance(_v, (int, float)):
                _out[_k] = _v
            else:
                _out[_k] = None if _v is None else type(_v).__name__
        return _out

    with open(_DIAG_STATE_DUMP, "a", encoding="utf-8") as _sf:
        _sf.write(
            _sj.dumps(
                {
                    "q": _state_summary(q_state),
                    "k": _state_summary(k_state),
                    "v": _state_summary(v_state),
                }
            )
            + "\\n"
        )

'''

RET = 'return {"q_state": q_state, "k_state": k_state, "v_state": v_state}'
CONST = "_REFINE_EDGE_EXTENSION = True"

jobs = [
    (r"logs\execution\diag_l21_offsets.py",
     r"logs\execution\diag_l21_state_oracle.py",
     r"logs\execution\diag_l21_state_oracle.jsonl"),
    (r"solutions\20260904_v186_attn-plus4-single-window_scoreNA_timeNA\solution.py",
     r"logs\execution\diag_l21_state_v186.py",
     r"logs\execution\diag_l21_state_v186.jsonl"),
]

for src, dst, dump_path in jobs:
    s = open(src, encoding="utf-8").read()
    assert s.count(RET) == 1, (src, s.count(RET))
    # insert call right before the return line, copying its indentation
    m = re.search(r"^([ \t]*)" + re.escape(RET), s, flags=re.M)
    indent = m.group(1)
    s = s.replace(indent + RET,
                  indent + "_diag_dump_states(q_state, k_state, v_state)\n"
                  + indent + RET, 1)
    # module-level helper + dump path constant
    anchor = "_DIAG_DUMP = " if "_DIAG_DUMP" in s else CONST
    i = s.index(anchor)
    line_end = s.index("\n", i)
    s = (s[: line_end + 1]
         + f'_DIAG_STATE_DUMP = r"{dump_path}"\n\n'
         + FUNC_CODE
         + s[line_end + 1:])
    io.open(dst, "w", encoding="utf-8", newline="\n").write(s)
    print("written", dst)
