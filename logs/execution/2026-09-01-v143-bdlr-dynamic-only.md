# v143 BDLR dynamic-only execution

v143 prevented BDLR from changing calibration-time W selection and used the rank-4 cross-block gradient only in the final dynamic activation call. The full run returned Linear `0.361153657663258`, Attention `0.7159419612310174`, API `207.44524579925928 s`, wall `230.78784280002583 s`.

This isolated the cause to the online update itself: the code remained time-safe, but the undamped column-only correction changed many legal activation codes and harmed output reconstruction. Representative-case sweeps show a small positive mean at damping strengths around `0.01–0.02`; v144 uses `0.02` without changing rank or call count.
