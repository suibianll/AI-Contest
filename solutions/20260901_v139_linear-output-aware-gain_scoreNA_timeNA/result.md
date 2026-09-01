# v139 result

- Parent: v138
- Change: replace the existing weight-only activation `output_gain` calibration
  with a cross-fold, block-64 output-aware diagonal Newton target. The deployed
  dynamic API keeps exactly the same single elementwise gain multiplication.
- Source SHA256: `8bdb2a100a51bd16301cd4392d4bbb68662bac03b531fc2643d6b0ee7f2c4cbd`
- Protocol: `official-shape-v1`, read-only Qwen cache, CUDA

| Linear mean | Attention mean | API total | Wall |
|---:|---:|---:|---:|
| 0.5072782560 | 0.7159419612 | 193.3892126 s | 217.1957354 s |

Linear regressed by `-0.0000412489` versus v138 while Attention remained
bit-identical. The continuous block surrogate therefore does not predict the
final discrete HiF4 output well enough. v139 is rejected and is not promoted to
the root. Official score/time remain unregistered.
