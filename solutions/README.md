# HiF4 Solution Archive

Root `solution.py` is the only active submission. Archived source files are immutable.

顺序实验索引见 [progressive candidate ledger](../logs/execution/2026-08-27-progressive-candidate-ledger.md)。

## 官方评测集修订（2026-08-29）

官方面板现为 **250 个 Linear case + 200 个 Attention case**，分数按全部
case 求和，因此分数与端到端时间都会高于旧口径。下表已把已确认的 v031、
v034、v051、v066 官方列更新为新版结果；其余历史官方列保留原提交时的旧口径，
不可与新版绝对值直接比较。新版时间限制为 **420s（7 分钟）**。

| Version | Date | Topic | Local Linear | Local Attention | Local Time | Official Score | Official Time | Delta | Status | Directory |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v000 | 2026-08-25 | v9 baseline | NA | NA | NA | ~9000+ | NA | NA | accepted | [archive](20260825_v000_v9-baseline_score9000plus_timeNA/) |
| v001 | 2026-08-26 | current baseline | NA | NA | NA | 10250 | 127s | NA | accepted | [archive](20260826_v001_current-baseline_score10250_time127s/) |
| v002 | 2026-08-26 | youxilee/hif4 v2.0 | 0.5668* | 0.3786* | NA | 15313 | 137s | NA | official-b0-closed | [archive](20260826_v002_youxilee-hif4_score15000plus_timeNA/) |
| v003 | 2026-08-27 | C1 A1 real Attention selector | 0.5668 | 0.4497 | CPU stage 54.72s | NA | NA | local +0.0712 Attention | local-champion | [archive](20260827_v003_a1-real-attention-local_scoreNA_timeNA/) |
| v004 | 2026-08-27 | C2 independent-segment CVaR | 0.5668 | 0.4155 | CUDA stage 26.68s | NA | NA | local -0.0342 causal | local-rejected | [archive](20260827_v004_c2-segment-cvar-local_scoreNA_timeNA/) |
| v005 | 2026-08-27 | C2a query-segment CVaR | 0.5668 | 0.4444 | CUDA stage 19.65s | NA | NA | local -0.0053 causal | local-rejected | [archive](20260827_v005_c2a-query-segment-cvar-local_scoreNA_timeNA/) |
| v006 | 2026-08-27 | C3 top-K 8×8 Linear quadratic | 0.5779 | 0.4497 | CPU stage 54.29s | NA | NA | local +0.0110 Linear | local-champion | [archive](20260827_v006_c3-topk-8x8-quadratic-local_scoreNA_timeNA/) |
| v007 | 2026-08-27 | C4 8×8 coverage 10% | 0.5788 | 0.4497 | CUDA stage 20.02s | NA | NA | local +0.0009 Linear | local-accepted-not-promoted | [archive](20260827_v007_c4-8x8-coverage10-local_scoreNA_timeNA/) |
| v008 | 2026-08-27 | C5 top-K 16×16 Linear quadratic | 0.5802 | 0.4497 | CPU stage 55.92s | NA | NA | local +0.0023 Linear | local-champion | [archive](20260827_v008_c5-topk-16x16-quadratic-local_scoreNA_timeNA/) |
| v009 | 2026-08-27 | C6 16×16 coverage 4% | 0.5808 | 0.4497 | CUDA stage 20.64s | NA | NA | local +0.0006 Linear | local-accepted-not-promoted | [archive](20260827_v009_c6-16x16-coverage4-local_scoreNA_timeNA/) |
| v010 | 2026-08-27 | C7 top-K 32×32 Linear quadratic | 0.5814 | 0.4497 | CUDA stage 21.99s | NA | NA | local +0.0012 Linear | local-accepted-not-promoted | [archive](20260827_v010_c7-topk-32x32-quadratic-local_scoreNA_timeNA/) |
| v011 | 2026-08-27 | C8 bounded 64×64 Linear quadratic | 0.5811 | 0.4497 | CUDA stage 23.55s | NA | NA | local +0.0009 Linear | local-accepted-not-promoted | [archive](20260827_v011_c8-topk-64x64-quadratic-local_scoreNA_timeNA/) |
| v012 | 2026-08-27 | C9 16×16 second sweep | 0.5804 | 0.4497 | CUDA stage 22.35s | NA | NA | local +0.0003 Linear | local-accepted-not-promoted | [archive](20260827_v012_c9-16x16-second-sweep-local_scoreNA_timeNA/) |
| v013 | 2026-08-27 | C10 wide activation quadratic | 0.5811 | 0.4497 | CPU stage 50.99s | 15799 | 144s | +486 vs v002 official | official-champion | [archive](20260827_v013_c10-wide-activation-quadratic_score15799_time144s/) |
| v014 | 2026-08-27 | C11 wide activation 8×8 residual | 0.5816 | 0.4497 | CPU stage 60.02s | NA | NA | local +0.0031 proj | local-champion | [archive](20260827_v014_c11-wide-activation-8x8-local_scoreNA_timeNA/) |
| v015 | 2026-08-27 | C12 wide activation 16×16 residual | 0.5817 | 0.4497 | CUDA stage 22.80s | NA | NA | local +0.0007 proj | local-accepted-not-promoted | [archive](20260827_v015_c12-wide-activation-16x16-local_scoreNA_timeNA/) |
| v016 | 2026-08-27 | C13 all-width activation 8×8 | 0.5862 | 0.4497 | CUDA stage 23.34s | NA | NA | local +0.0046 Linear; amax4 o -0.0091 | local-accepted-not-promoted | [archive](20260827_v016_c13-all-width-activation-8x8-local_scoreNA_timeNA/) |
| v017 | 2026-08-27 | C14 gated all-width activation 8×8 | 0.5861 | 0.4497 | CPU stage 58.05s | NA | NA | local +0.0045 Linear | local-champion | [archive](20260827_v017_c14-gated-all-width-activation-8x8-local_scoreNA_timeNA/) |
| v018 | 2026-08-27 | C15 quantized-weight activation Gram | 0.5861 | 0.4497 | CUDA stage 25.62s | NA | NA | local ~0.0000 Linear | local-accepted-not-promoted | [archive](20260827_v018_c15-quantized-weight-activation-gram-local_scoreNA_timeNA/) |
| v019 | 2026-08-27 | C16 gated activation 8×8 coverage 4% | 0.5876 | 0.4497 | CUDA stage 24.78s | NA | NA | local +0.0015 Linear | local-accepted-not-promoted | [archive](20260827_v019_c16-gated-activation-8x8-coverage4-local_scoreNA_timeNA/) |
| v020 | 2026-08-27 | C17 final gated activation 8×8 coverage 8% | 0.5890 | 0.4497 | CPU stage 63.96s | NA | NA | local +0.0029 Linear | local-champion | [archive](20260827_v020_c17-final-gated-activation-8x8-coverage8-local_scoreNA_timeNA/) |
| v021 | 2026-08-27 | C18 activation/weight-error cross term | 0.5897 | 0.4497 | CUDA stage 25.19s | NA | NA | local +0.0008 Linear | local-accepted-not-promoted | [archive](20260827_v021_c18-activation-cross-term-local_scoreNA_timeNA/) |
| v022 | 2026-08-27 | C19 cross-aware gain selection | 0.5905 | 0.4497 | CUDA stage 25.25s | NA | NA | local +0.0015 Linear | local-accepted-not-promoted | [archive](20260827_v022_c19-cross-aware-gain-selection-local_scoreNA_timeNA/) |
| v023 | 2026-08-27 | C20 exact discrete cross-gain selection | 0.5931 | 0.4497 | CUDA stage 25.19s | 16081 | 152s | historical non-compliant anchor (output supervision fed Q(A)) | official-submitted-non-compliant | [archive](20260827_v023_c20-exact-discrete-cross-gain-local_scoreNA_timeNA/) |
| v024 | 2026-08-27 | C21 gated exact cross selection | 0.5930 | 0.4497 | CUDA stage 25.87s | 16043 | 173.8s | +244 vs v013 official | official-champion | [archive](20260827_v024_c21-gated-exact-cross-selection_score16043_time174s/) |
| v025 | 2026-08-27 | C21-C compliant baseline | 0.5311 | 0.4497 | CPU stage ~61.3s | 14437 | 166.6s | compliant official anchor | official-compliant-anchor | [archive](20260827_v025_c21c-compliance-baseline/) |
| v026 | 2026-08-27 | C22 Linear R64 | 0.5311 | 0.4497 | local ratio 1.52 | NA | NA | all components fell back | local-rejected | [archive](20260827_v026_c22-linear-r64-rejected_scoreNA_timeNA/) |
| v027 | 2026-08-27 | C23 FULL64 Weight | 0.5504 | 0.4497 | local ratio 1.55 | NA | NA | local +1.93pp Linear | archived-rejected; mechanism later promoted | [archive](20260827_v027_c23-full64-rejected_scoreNA_timeNA/) |
| v028 | 2026-08-27 | activation scale-code oracle | NA | NA | probe only | NA | NA | no submission source | diagnostic-only | [archive](20260827_v028_c28-scale-code-probe-rejected/) |
| v029 | 2026-08-28 | C29 HAES probe | 0.5311 | 0.4497 | probe only | NA | NA | no active behavior change | local-rejected | [archive](20260828_v029_c29-haes-rejected_scoreNA_timeNA/) |
| v030 | 2026-08-28 | C38 beam2 + narrow FULL64 full-coverage | 0.5695 | 0.4497 | CUDA stage ~30s | 14092 | 170.57s | official inversion vs local +3.84pp (pending A/B) | official-submitted, local-champion, not-promoted | [archive](20260828_v030_c38-beam2-fullcov-official14092_time170.6s/) |
| v031 | 2026-08-28 | C39-FW wide-layer FULL64 calibration candidate | 0.5357 | 0.4497 | CUDA stage 27.47s | **21864** | **161.3s** | **新版面板锚点；与 v034 同分** | **official-compliant-anchor** | [archive](20260828_v031_c39-fw-official14613_time159.2s/) |
| v032 | 2026-08-28 | C40 robust Block-LDLQ 128 | 0.5393 | 0.4497 | CUDA stage 45.32s; CPU 100.05s | **14432** | **216.667s** | **旧面板 −181 vs 旧 v031；与新版不直接比较** | official-rejected | [archive](20260828_v032_c40-robust-blockldlq_official-score14432_time216.667s/) |
| v033 | 2026-08-29 | C41 scale-aware K 公共平移（Attention） | 0.5357（Linear 逐位不变） | MHA +0.72% / +0.75%；GQA −0.88% | API 最慢 74.48s | NA | NA | 总分 −0.074；MHA 正向、GQA 负向 | local-rejected | [archive](20260829_v033_c41-scale-aware-k-center_scoreNA_timeNA/) |
| v034 | 2026-08-29 | C41b scale-aware K 中心（仅 MHA，GQA 禁用） | 0.5357（Linear 逐位不变） | MHA +0.72% / +0.75%；GQA 0% | API 最慢 70.71s | **21864** | **159.4s** | **新版面板与 v031 同分，快 1.9s** | **official-compliant-anchor** | [archive](20260829_v034_c41b-mha-k-center_scoreNA_timeNA/) |
| v035 | 2026-08-29 | C42e calibration-product compensation | 仅 GPT-2 small 局部代理 `130.183032` | `21.120464` | 35.181s | NA | NA | 高维校准过拟合风险 | **archived-rejected** | [archive](20260829_v035_c42e-product-compensation-rejected_scoreNA_timeNA/) |
| v036 | 2026-08-29 | C43 analytic CAT-64 | `128.441940` | `21.120464` | 67.115s | NA | NA | Linear `-0.901369` vs C41b；Attention 逐位相同 | **archived-rejected** | [archive](20260829_v036_c43-cat64-rejected_scoreNA_timeNA/) |
| v037 | 2026-08-29 | C43b CAT-64 β=0.25 | `130.939221` | `21.120464` | 39.740s | NA | NA | Linear `+1.595912` vs C41b；GPT-2 small positive | **local-accepted** | [archive](20260829_v037_c43b-cat64-beta025_scoreNA_timeNA/) |
| v038 | 2026-08-29 | C43c CAT-64 full-H selector | GPT-2 `131.349623`; OPT `-158.429511`; Qwen `258.554132` | unchanged | 39–109s | NA | NA | full-H proxy unstable and model-sensitive | **archived-rejected** | [archive](20260829_v038_c43c-fullh-rejected_scoreNA_timeNA/) |
| v039 | 2026-08-29 | C45b fixed-Q(A) A@W static Q(W) selector | GPT-2 `129.712444`; OPT `29.862380`; Qwen `269.025229` | `21.120464` / `19.581565` / `62.862350` | 45.76–122.64s | NA | NA | 三模型均低于 C43b，固定 Q(A) 产品目标过拟合 | **archived-rejected** | [archive](20260829_v039_c45b-fixed-qactivation-rejected_scoreNA_timeNA/) |
| v040 | 2026-08-29 | C45c 原始 A@W 静态 Q(W) + max-dim 4096 | GPT-2 `131.769809`; OPT `31.602006`; Qwen `286.174039` | `21.120464` / `19.581565` / `62.862350` | 45.70–103.99s | NA | NA | 三模型合计较 C43b `+0.542193`；Qwen 宽层回退得到控制 | **local-accepted** | [archive](20260829_v040_c45c-raw-sizecapped_scoreNA_timeNA/) |
| v041 | 2026-08-29 | C44 MR-GPTQ parent full-H 覆盖 97% | GPT-2 `126.219696` | `21.120464` | 39.72s（partial） | NA | NA | GPT-2 较 C43b `-4.719525`；扩大 coverage 扩散误差 | **archived-rejected** | [archive](20260829_v041_c44-mr-gptq-coverage97-rejected_scoreNA_timeNA/) |
| v042 | 2026-08-29 | C45e 多折 A@W 静态 Q(W) + max-dim 4096 | GPT-2 small `133.226930`; medium `229.019937`; OPT `32.580090`; Pythia `138.246673`; Qwen `286.174039` | `21.120464` / `43.767156` / `19.581565` / `40.614368` / `62.862350` | 45.20–104.70s | NA | NA | 五模型合计较 C43b `+5.188928`；全部不低于父版本 | **local-accepted** | [archive](20260829_v042_c45e-product-allfolds-sizecap_scoreNA_timeNA/) |
| v043 | 2026-08-29 | C45f adaptive headroom `{-4..4}` + 多折 A@W | GPT-2 small `133.226930`; medium `229.019937`; OPT `43.279017`; Pythia `138.246673`; Qwen `286.174039` | `21.120464` / `43.767156` / `19.581565` / `40.614368` / `62.862350` | 46.58–115.68s | NA | NA | 五模型合计较 C43b `+15.887855`；OPT +10.698927 | **local-accepted** | [archive](20260829_v043_c45f-headroom-multifold_scoreNA_timeNA/) |
| v044 | 2026-08-29 | C46a CAT β `{0.125,0.25,0.375}` | GPT-2 small `133.601982`; medium `229.271482`; Qwen `286.499658`; OPT `-846.212506` | 对应 `21.120464` / `43.767156` / `62.862350` / `19.581565` | 63.88–166.17s（partial） | NA | NA | OPT 结构性回退 `-826.630941`，β 网格 rejected | **archived-rejected** | [archive](20260829_v044_c46a-cat-beta-grid-rejected_scoreNA_timeNA/) |
| v045 | 2026-08-29 | C45g 放开 Qwen 4864-wide headroom | Qwen `286.174039` | `62.862350` | 120.87s | NA | NA | 分数与 C45f 相同，仅增加校准开销 | **archived-rejected** | [archive](20260829_v045_c45g-headroom-qwenwide-rejected_scoreNA_timeNA/) |
| v046 | 2026-08-29 | C46b CAT β 窄 refinement `{0.20,0.25,0.30}` | OPT `31.214825` | `19.581565` | 67.81s | NA | NA | OPT Total `50.796390`，较 C45f `-12.064192` | **archived-rejected** | [archive](20260829_v046_c46b-cat-beta-narrow-rejected_scoreNA_timeNA/) |
| v047 | 2026-08-29 | C45h 全宽多折 A@W 产品选择，预算 8192 | Qwen `285.702496` | `62.862350` | 131.03s | NA | NA | Qwen Total `348.564846`，较 C45f `-0.471543`；4864-row FFN 回退 | **archived-rejected** | [archive](20260829_v047_c45h-product-allfolds-qwen-rejected_scoreNA_timeNA/) |
| v048 | 2026-08-29 | C45i 按输出行数限制静态 A@W 产品选择 | GPT-2 small `133.226930`; medium `229.019937`; OPT `43.279017`; Pythia `138.246673`; Qwen `286.266123` | `21.120464` / `43.767156` / `19.581565` / `40.614368` / `62.862350` | 44.21–124.82s | NA | NA | 五模型合计 `1017.984583`，较 C45f `+0.092084`；其余四模型持平 | **local-accepted** | [archive](20260829_v048_c45i-product-outputrowcap_scoreNA_timeNA/) |
| v049 | 2026-08-29 | C49 CAT block-Hessian operand metric | GPT-2 small `133.226930`; medium `229.019937`; Qwen `286.266123` | `21.120464` / `43.767156` / `62.862350` | 50.35–128.59s | NA | NA | 三模型逐项持平 v048，仅增加校准开销 | **archived-rejected** | [archive](20260829_v049_c49-cat-block-hessian-rejected_scoreNA_timeNA/) |
| v050 | 2026-08-29 | C47 CAT-aware 4→64 channel grouping | GPT-2 small `133.226930`; medium `229.041484`; OPT `42.981262`; Pythia `138.382957`; Qwen `286.474705` | `21.120464` / `43.767156` / `19.581565` / `40.614368` / `62.862350` | 53.87–156.89s | NA | NA | 五模型合计 `1018.053241`，较 v048 `+0.068658`；OPT −0.297755 | **local-accepted** | [archive](20260829_v050_c47-cat-grouping_scoreNA_timeNA/) |
| v051 | 2026-08-29 | C47b grouping 0.5% soft gate | GPT-2 small `133.226930`; medium `229.041484`; OPT `43.279017`; Pythia `138.329016`; Qwen `286.481992` | `21.120464` / `43.767156` / `19.581565` / `40.614368` / `62.862350` | 55.67–149.00s | **22451** | **234s** | **新版面板较 v031/v034 +587 分；仍低于 420s** | **official-compliant-champion** | [archive](20260829_v051_c47b-grouping-threshold005_scoreNA_timeNA/) |
| v052 | 2026-08-29 | C47c grouping 1% soft gate | GPT-2 small `133.226930`; medium `229.019937`; OPT `43.279017`; Pythia `138.246673`; Qwen `286.481992` | `21.120464` / `43.767156` / `19.581565` / `40.614368` / `62.862350` | 52.64–148.29s | NA | NA | 五模型合计 `1018.200452`，较 v048 `+0.215869`，低于 v051 | **archived-rejected** | [archive](20260829_v052_c47c-grouping-threshold01-rejected_scoreNA_timeNA/) |
| v053 | 2026-08-29 | C48 CAT + 16/32-channel micro-Hadamard | GPT-2 small `133.226930`; Qwen `286.481992`; OPT `43.279017` | `21.120464` / `62.862350` / `19.581565` | 53.83–150.69s | NA | NA | 三模型逐项持平 v051，未接受任何组合 | **archived-rejected** | [archive](20260829_v053_c48-micro-hadamard-rejected_scoreNA_timeNA/) |
| v054 | 2026-08-29 | C54 Weight headroom 覆盖 50% | GPT-2 small `133.226930`; medium `229.041484`; OPT `48.008010`; Pythia `138.329016`; Qwen `286.481992` | `21.120464` / `43.767156` / `19.581565` / `40.614368` / `62.862350` | 53.39–149.42s | NA | NA | 五模型合计 `1023.033335`，较 v051 `+4.728993`；OPT +4.728993 | **local-accepted** | [archive](20260829_v054_c54-headroom50_scoreNA_timeNA/) |
| v055 | 2026-08-29 | C55 Weight headroom 覆盖 75% | GPT-2 small `133.226930`; medium `229.041484`; OPT `49.823139`; Pythia `138.329016`; Qwen `286.481992` | `21.120464` / `43.767156` / `19.581565` / `40.614368` / `62.862350` | 56.48–155.05s | NA | NA | 五模型合计 `1024.848464`，较 v054 `+1.815129`；OPT +6.544122 vs v051 | **local-accepted** | [archive](20260829_v055_c55-headroom75_scoreNA_timeNA/) |
| v056 | 2026-08-29 | C56 Weight headroom 覆盖 100% | GPT-2 small `133.226930`; medium `229.148763`; OPT `50.307533`; Pythia `138.329016`; Qwen `286.481992` | `21.120464` / `43.767156` / `19.581565` / `40.614368` / `62.862350` | 56.36–156.16s | NA | NA | 五模型合计 `1025.440137`，较 v055 `+0.591673`；较 v051 `+7.135795` | **local-accepted** | [archive](20260829_v056_c56-headroom100_scoreNA_timeNA/) |
| v057 | 2026-08-29 | C57 静态 A@W 产品候选比例 25% | OPT `49.987750`; Qwen `286.032560`; GPT-2 medium `228.673780` | `19.581565` / `62.862350` / `43.767156` | 62.12–171.52s | NA | NA | 三模型均较 v056 回退（−0.319782/−0.449432/−0.474983） | **archived-rejected** | [archive](20260829_v057_c57-product-ratio25-rejected_scoreNA_timeNA/) |
| v058 | 2026-08-29 | C58 Headroom E6M2 offsets `{-6,…,6}` | OPT `50.307481`; Qwen `286.481992` | `19.581565` / `62.862350` | 54.18–154.15s | NA | NA | 两模型与 v056 持平（OPT −5.2e-5），未形成有效候选 | **archived-rejected** | [archive](20260829_v058_c58-headroom-offsets6-rejected_scoreNA_timeNA/) |
| v059 | 2026-08-29 | C59 逐 64-block A@W headroom 混合 | OPT `35.639634` | `19.581565` | 54.42s | NA | NA | OPT Total `55.221199`，较 v056 `−14.667899`，严重过拟合 | **archived-rejected** | [archive](20260829_v059_c59-headroom-blockwise-rejected_scoreNA_timeNA/) |
| v060 | 2026-08-29 | C60 A@W 产品条件步长 `{0.05,…,0.75}` | OPT `48.878251` | `19.581565` | 57.82s | NA | NA | OPT Total `68.459816`，较 v056 `−1.429282`，候选自由度过高 | **archived-rejected** | [archive](20260829_v060_c60-product-alpha-grid-rejected_scoreNA_timeNA/) |
| v061 | 2026-08-29 | C61 CAT `WᵀW` 统计 1024 行 | GPT-2 small `133.668200`; OPT `50.271056`; Qwen `266.266543` | `21.120464` / `19.581565` / `62.862350` | 54.45–151.55s | NA | NA | GPT-2 small `+0.441270` 但 Qwen Total `329.128893`，较 v056 `−20.215449` | **archived-rejected** | [archive](20260829_v061_c61-cat-weightgram1024-rejected_scoreNA_timeNA/) |
| v062 | 2026-08-29 | C62 CAT `WᵀW` 宽度分流（≤4096 用 1024 行） | GPT-2 small `133.668200`; medium `229.310238`; OPT `50.271056`; Pythia `138.198315`; Qwen `286.860477` | `21.120464` / `43.767156` / `19.581565` / `40.614368` / `62.862350` | 54.28–152.35s | NA | NA | 五模型合计 `1026.254189`，较 v056 `+0.814052`；Qwen 回升 `+0.378485` | **local-accepted** | [archive](20260829_v062_c62-cat-weightgram-widthcap_scoreNA_timeNA/) |
| v063 | 2026-08-29 | C63 Linear 候选 `weight_sample` 512 行 | GPT-2 small `134.316520`; medium `231.081080`; OPT `65.463078`; Pythia `138.282530`; Qwen `287.007536` | `21.120464` / `43.767156` / `19.581565` / `40.614368` / `62.862350` | 54.06–151.69s | NA | NA | 五模型合计 `1044.096647`，较 C62 `+17.842458`；五模型全正向 | **local-accepted** | [archive](20260829_v063_c63-linear-weight-eval512_scoreNA_timeNA/) |
| v064 | 2026-08-29 | C64 Linear 候选 `weight_sample` 1024 行 | GPT-2 small `134.064366`; OPT `63.119458`; Qwen `287.281990` | `21.120464` / `19.581565` / `62.862350` | 54.80–155.17s | NA | NA | 三模型小计较 C63 `−2.321320`；512 行更稳 | **archived-rejected** | [archive](20260829_v064_c64-linear-weight-eval1024-rejected_scoreNA_timeNA/) |
| v065 | 2026-08-29 | C65 A@W 折间软混合 `0.50` | GPT-2 small `134.255521`; OPT `65.463078` | `21.120464` / `19.581565` | 55.24–55.80s | NA | NA | OPT 与 C63 持平，GPT-2 small `−0.0610`；`0.25` 更稳 | **archived-rejected** | [archive](20260829_v065_c65-product-robustmix50-rejected_scoreNA_timeNA/) |
| v066 | 2026-08-29 | C66 动态激活损失覆盖目标 `1.0` | GPT-2 small `134.327340`; medium `231.098280`; OPT `65.472279`; Pythia `138.290983`; Qwen `287.032704` | `21.306236` / `43.760024` / `19.647602` / `40.647879` / `63.119717` | 54.29–151.91s | **22557** | **217.2s** | **新版面板本地归档冠军；较 v051 +106 分、快 16.8s** | **official-compliant-champion** | [archive](20260829_v066_c66-activation-ratio100_scoreNA_timeNA/) |
| v067 | 2026-08-29 | C67 Linear 候选 `weight_sample` 640 行 | GPT-2 small `134.335127`; medium `230.953064`; OPT `65.687656`; Pythia `138.183341`; Qwen `287.110685` | `21.306236` / `43.760024` / `19.647602` / `40.647879` / `63.119717` | 54.00–155.95s | NA | NA | 五模型合计 `1044.751331`，仅较 C66 `+0.048287`；medium/Pythia 回退 | **archived-rejected** | [archive](20260829_v067_c67-linear-weight-eval640-rejected_scoreNA_timeNA/) |
| v068 | 2026-08-29 | C68 A@W 静态块预算 `15%` | GPT-2 small `134.416424`; OPT `65.246115`; Qwen `286.922986` | `21.306236` / `19.647602` / `63.119717` | 57.98–159.93s | NA | NA | 三模型小计较 C66 `−0.246797`；12.5% 更稳 | **archived-rejected** | [archive](20260829_v068_c68-product-ratio15-rejected_scoreNA_timeNA/) |
| v069 | 2026-08-29 | C69 激活二次项 Gram-8 覆盖上限 `12%` | GPT-2 small `134.329831`; medium `231.098280`; OPT `65.472699`; Pythia `138.291865`; Qwen `287.032704` | `21.306236` / `43.760024` / `19.647602` / `40.647879` / `63.119717` | 54.95–156.68s | NA | NA | 五模型合计 `1044.706838`，较 C66 `+0.003794`；全模型非负 | **local-accepted** | [archive](20260829_v069_c69-activation-gram8-ratio12_scoreNA_timeNA/) |
| v047 | 2026-08-29 | C45h 全宽多折 A@W 产品选择，预算 8192 | Qwen `285.702496` | `62.862350` | 131.03s | NA | NA | Qwen Total `348.564846`，较 C45f `-0.471543`；4864-row FFN 回退 | **archived-rejected** | [archive](20260829_v047_c45h-product-allfolds-qwen-rejected_scoreNA_timeNA/) |

## 外部参考（不纳入本地版本号）

| 代码 | 新版官方分数 | 新版官方时间 | 说明 |
| --- | ---: | ---: | --- |
| [`youxilee/hif4`](https://github.com/youxilee/hif4) | **24153** | **239s** | 用户提供的结果；代码未导入、未在本地复现 |

该外部结果比本地 v066/C66 高 `1596` 分、多 `21.8s`；它是后续 Linear
结构优化的参考上限，不改变本地候选的可复现状态。分数和时间为用户提供的
同口径结果，仓库页面本身未给出可独立核验的官方排行榜记录。

`*` v002 的 Linear/Attention 数值最初来自远程仓库 `CHANGELOG.md` 的 GPT-2
12 层、2 calib + 2 test 报告，之后已由 GPU-compatible B0 derivative 在本地
复现。该 derivative 与 v002 归档行为等价但 SHA256 不同，因此表中 Local Time
仍为 `NA`。用户于 2026-08-27 确认 v002/B0 的官方结果为 `15313 / 137s`，
B0 据此闭环；本地 derivative SHA 不作为官方上传文件 SHA。

v003 起允许只有本地结果时立即归档。未提交候选的官方列保持 `NA`，未来结果返回时追加
提交 SHA、分数和时间，不覆盖既有本地配对表或实验结论。

用户于 2026-08-27 确认 v013（C10，提交 `a2e0ed3`）官方结果为 `15799 / 144s`，
较 v002/B0 的 `15313 / 137s` 提升 `+486`；该提交的 `solution.py` 经 git blob 校验与
v013 归档字节一致。

用户于 2026-08-27 确认 v024（C21，提交 `23d1cf7`）官方结果为
`16043 / 173.8s`。该版本的 Linear 输出监督把 `A@W` 信息用于激活侧选择，
因此在当前规则下不合规，只保留为历史官方记录。离线校准中用 `A@W`
优化 `Q(W)` 本身是允许的。v025 / C21-C 是最新合规官方锚点：
`14437 / 166.6s`。

2026-08-29 官方评测集扩大为 250 个 Linear case 与 200 个 Attention case，
并将时间限制提升至 7 分钟。用户确认新版官方结果：v031/C39-FW 为
`21864 / 161.3s`，v034/C41b 为 `21864 / 159.4s`，v051/C47b 为
`22451 / 234s`，v066/C66 为 `22557 / 217.2s`；这些数值覆盖对应条目的
旧版 `Official Score/Time`，旧值仍可在各自提交历史中追溯。

当前根 `solution.py` 为 C69 本地候选；C69 归档源码的 SHA256 为
`1F71CA11FA9707EB9720438EC6D780CC6F520FBA80437B3215398608D5866CA1`。
新版面板下 v066/C66（官方 `22557 / 217.2s`）是本地归档冠军，较此前
v051/C47b 提升 `106` 分并减少 `16.8s`；v031/C39-FW 与 v034/C41b 均为
`21864`。外部 [`youxilee/hif4`](https://github.com/youxilee/hif4) 的
`24153 / 239s` 仍高出 `1596` 分，仅作参考，不以根文件位置暗示外部代码已导入。

## Local-first workflow

1. Modify only the root `solution.py`; archived sources remain immutable.
2. Run the real-GPT evaluator:

   ```powershell
   .\.venv\Scripts\python evaluator\real_data_eval.py --solution solution.py --model gpt2
   ```

3. Record the six Linear components, Attention results, evaluator parameters, and paired local runtime.
4. Coherent compute reallocation may change several coupled knobs together; record the ablations afterward.
5. Preserve useful checkpoints and rejected mechanisms in the next immutable `vNNN` directory.
6. Use `scoreNA_timeNA` while official evaluation is unavailable; copy the exact evaluated source and verify SHA256 equality.
7. Do not require a fixed minimum gain for exploration; map the accuracy-runtime Pareto and keep stable improvements.
8. Treat oracle/proxy results as diagnostics, not as substitutes for deployed-path evaluation.
9. Before submission, enforce compliance, legal state/API behavior, and the official `<420s` runtime limit (7 minutes).
10. If official results later become available, append the submitted SHA, score, runtime and date without overwriting local evidence.

Use `NA`, `score9000plus`, or `time300plus` when a historical value is unavailable, approximate, or timed out. Never replace an unknown official value with a local estimate.
