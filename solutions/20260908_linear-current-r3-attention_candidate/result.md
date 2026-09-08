# Current Linear + R3 Attention — local candidate

## Composition

- Linear: current root compiled sample-energy path, root SHA `D66128A6...B0F6`.
- Attention: R3 rotation-center stack, source SHA `A5C679D7...146DC`.
- Candidate SHA: `12352EFDD4E23CC5E1E17953008664FBAA5EA5D693373635FDAFC4D28CE4E24E`.

The candidate is a separate archive and does not replace the current root. The
R3 file's side-isolation Linear shadow definitions are overridden at the end
with the current root's compiled sample-energy calibration and dynamic path;
R3's four Attention APIs and learned-state calibration remain the final
Attention definitions.

## Expected score and time

The additive estimate is `18032`:

`17636 + (14405 - 14009) = 18032`.

This uses the historically measured additive side model and is not an official
score prediction. The time risk estimate is approximately `291s` from
`264 + (238 - 211)`; this is only a prioritization signal, not a local time
gate. Official score and the 300s hard limit remain the only promotion tests.

## Verification state

Six public API import, random-shape execution, finite-output and reference
state validation passed. A proxy-v3 4B shard-0 `both` run completed with
`reasonableness_issues=0`.

The shard-0 paired delta was Linear `0.000000` (56/56 zero cases) and
Attention `+0.017241` mean, `0.017574` L1, with 11 positive, 1 negative and
0 zero cases. The analyzer therefore rejects the candidate for this screen
because Linear has no incremental effect; the Attention gain is positive but
mixed. Local API time was `187.24s` for this shard and is diagnostic only.

The candidate remains `UNREGISTERED/NA`; no official score or runtime is
claimed here, and the current root is not replaced.
