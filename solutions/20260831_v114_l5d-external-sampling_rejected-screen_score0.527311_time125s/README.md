# v114 L5d external sampling candidate — rejected at screen

- Parent: v111 L5a joint-permutation precision parent
- Change: replace root `_sample_rows` linspace/round selection with external hif4
  `step=ceil(rows/limit); x[::step][:limit]`; no other code or state changes.
- Candidate LF SHA256: `4b6fa66827eabaeebbdf5289ba0c936aab43e610b36981527c402b76a2d617d5`
- Screen: Qwen2.5-0.5B, layers `0,5,11,17,23`, roles `q,k,v,o,fc_gate,fc_up,proj`,
  fixed cache, CPU, 125.332 s.
- Linear mean: `0.5273114999151473`
- Parent screen Linear mean: `0.5318869456762372`
- Difference: `-0.0045754457610899`
- Decision: rejected; no full-layer run. Root restored to v111.

Files in this archive are the exact candidate source and its screen JSON/log.
