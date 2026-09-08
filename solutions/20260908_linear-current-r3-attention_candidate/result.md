# Current Linear + R3 Attention — official retained

## Composition

- Linear: current root compiled sample-energy path, root SHA `D66128A6...B0F6`.
- Attention: R3 rotation-center stack, source SHA `A5C679D7...146DC`.
- Candidate SHA: `12352EFDD4E23CC5E1E17953008664FBAA5EA5D693373635FDAFC4D28CE4E24E`.

The archived source is now the current root. The R3 file's side-isolation
Linear shadow definitions are overridden at the end with the previous root's
compiled sample-energy calibration and dynamic path; R3's four Attention APIs
and learned-state calibration remain the final Attention definitions.

## Official result

The user reported an official score/time of `18032 / 280s`, which is `+396 / +16s`
versus the previous root `17636 / 264s`. The score is above the previous root
and the time is below the official `300s` hard limit, so the source is retained
as the current working parent.

The earlier additive diagnostic was:

`17636 + (14405 - 14009) = 18032`.

It happened to match the official score; the official result takes precedence
over that diagnostic. The platform did not return a separate scored SHA, so the
result is transparently bound to the archived candidate SHA above.

## Verification state

Six public API import, random-shape execution, finite-output and reference
state validation passed. A proxy-v3 4B shard-0 `both` run completed with
`reasonableness_issues=0`.

The shard-0 paired delta was Linear `0.000000` (56/56 zero cases) and
Attention `+0.017241` mean, `0.017574` L1, with 11 positive, 1 negative and
0 zero cases. The analyzer therefore rejects the candidate for this screen
because Linear has no incremental effect; the Attention gain is positive but
mixed. Local API time was `187.24s` for this shard and is diagnostic only.

The source is retained as `CURRENT_ROOT`; the previous root remains recoverable
through git history and its archived source directory.
