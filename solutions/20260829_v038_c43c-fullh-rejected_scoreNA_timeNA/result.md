# C43c CAT-64 full-H selection（归档）

- 日期：2026-08-29
- 版本：v038 / C43c
- 父版本：v037 / C43b
- 唯一变化：CAT 选择器增加 64×64 activation block full-H weight proxy，并将 objective 权重调整为 `0.60 weight + 0.30 activation + 0.10 alignment`。
- A@W：未使用。
- 根文件 SHA256：`F97403AB0A57E8D5C472F0E317DC96AADA80119C8F110EDB79B6500B119175B3`
- 归档文件 SHA256：`F97403AB0A57E8D5C472F0E317DC96AADA80119C8F110EDB79B6500B119175B3`

## 评测

- GPT-2 small：Total `152.470087` / Linear `131.349623` / Attention `21.120464` / API `39.61s`。
- OPT-125M：Total `-138.847946` / Linear `-158.429511` / Attention `19.581565` / API `38.17s`。
- Qwen2.5-0.5B：Total `321.416482` / Linear `258.554132` / Attention `62.862350` / API `109.29s`。
- 评测命令和缓存协议见 `logs/evaluations/2026-08-29-c43c-3model.md`。

## 结论

full-H proxy 与现有 FULL64 solver 的实际行为仍不一致，且出现明显模型/调用顺序敏感性；该版本拒绝。根文件已恢复到 v037 C43b。
