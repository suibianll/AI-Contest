# v147 official result correction

Date: 2026-09-01

The user confirmed the following official result:

- Version: v147
- Official score: **16579**
- Official runtime: **211s**
- Runtime outcome: pass (`<300s`)
- Final repository decision: **REJECTED**

v147 is rejected because its official score is below both the verified v86 baseline
(`16744 / 222.7s`) and the user-confirmed high score 17816. Passing the runtime limit does not make
an inferior score a retained parent.

The official submitted source SHA is unconfirmed. The v147 archive was previously modified in
place: the original local JSON records SHA `9B3EA5...B656` and Linear `0.5073546371`, while the later
direct-merge A3 JSON records SHA `25C245...9C1B` and Linear `0.5100503237`; the current cleaned source
has SHA `44E377...2672`. The official result is therefore recorded independently and is not bound to
any of these SHAs without additional user evidence.

No raw JSON or local report was rewritten. The solution directory was renamed to
`20260901_v147_v86-attention-v140-linear_rejected`, and the active plan continues from a restored,
reproducible pre-A3 control rather than v147 as a retained parent.
