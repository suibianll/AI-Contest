# A-JC1：输入条件化码修正与完整参数曲率

状态：合同/CPU验证已启动；未构建正式候选，未运行4B。父v237完整根，SHA见contract-results.json。

## 固定部署规则

只在父Q/K编码全部refine完成后修改mant/sign；冻结scale_factor/lv2/lv3、父变换、K center、V和Linear。零码冻结，不插入新符号。不能先改输入再让旧编码器重选scale，避免修正被层级吸收。

父最终整数mant为m=4*mant，单码步长s=scale_factor*lv2*lv3/4；x为父变换完成后的真实量化输入，e=clip(abs(x)/s-m,-1,1)。每元素特征固定为phi=[1,e,m/7,mean_4(e)]，最后一项是同一合法4元素子组平均，不跨token/head。每层theta_q与theta_k各4个参数，共8个；层间使用同一算法，不做特殊层路由。

动态一次计算c=round(clip(phi theta,-1,1))，输出m'=clip(m+c,0,7)。父sign=0时m'=0；m'=0时sign'=0，其余保留父sign。所有字段合法、theta=0严格返回父。Q只读自己的输入与theta_q，K同理，无动态QK/Gram/候选循环。实际实现需在父变换至编码之间返回局部x，不使用模块全局开关或跨API缓存。

## 校准求解

固定fit窗口0/1/2，各等权；selection窗口3/4，保持两个窗口均严格改善的父/单候选接受规则。独立validation只报告。少于5个窗口的contract输入原样回退，不人为补窗。GQA按真实head共享关系聚合，V固定使用父解码值。

先为8个连续参数构建解析输出方向：B_role=sign*s*phi；deltaQ=Bq theta_q、deltaK=Bk theta_k。以父Z=Q0 K0^T/sqrt(d)为锚，A=softmax(mask(Z))；每列deltaA=A*(deltaZ-sum(A*deltaZ))，J_col=deltaA V0。该J属于平滑松弛，不是硬round导数。按输出元素mean及fold等权累计8×8 H=J^T J、g=J^T(O0-Oref)，保留Q/K交叉块。阻尼lambda=max(1e-4*trace(H)/8,1e-12)，float64解一次(H+lambda I)theta=-g；不扫阻尼或放大theta。

这一阶J不包含二阶deltaQ deltaK项，完整hard forward包含该项和softmax非线性。不能宣称完整Hessian或精确二次目标；这是8维完整Gauss-Newton矩阵。不存储全token-channel Hessian；逐输出块/窗口累计H,g，按真实mask检查完整与分块一致性。

首轮只产生一个theta，不增加步数、seed、阈值搜索或线搜索。报告平滑预测、实际hard fit、selection、硬码改动与拒绝原因。低于0.8不阻止探索；全回退不提交。若需新的硬离散求解器，明确另写实质算法修订，不偷偷调系数重跑。

## 和A-RB1的实质差别

旧A-RB1为码级12个共享阈值、单Hutchinson探针对角曲率、64桶分别argmin，并重选层级；记录显示联合翻码恶化。A-JC1是输入条件化的8系数规则、完整参数空间交叉曲率、一次联合解、冻结最终父层级。不是增加阈值数量或换seed。旧失败保留，新方法是否足够跨越硬边界尚未证明。

## 已执行的首个实验

contract_smoke.py只在CPU合成8 token、2Q/1KV head、head_dim64运行：零参数逐字段一致、强制参数可达、五字段合法、scale不变、平滑JVP有限差分相对误差1.32e-10；Q/K交叉块非零。

但第一次固定求解：代理MSE 0.00443685→0.00321070，真实hard MSE变为0.00459145，Q/K分别改20/23个码。**接口与推导通过，精度尚未通过验证；不能用代理下降宣称算法有效。** 不把单个合成负例当4B/官方否决；下一步先验证真实父路径x/scale坐标及mask，再做单层校准诊断，分别记录连续预测与hard误差，不直接启动六shard。

## 立即执行顺序

1. 绑定最终生效Q/K编码调用图，接出与最终父scale对应的x；零theta在真实父initial/refine两路径逐字段一致，验证state CPU/stride语义。
2. 用已完成同口径报告绑定当前Attention基线；如SHA不同，只能静态证明Attention调用图等价后复用，不能直接继承整包结果。
3. 选择manifest首个真实Attention层做一次固定fit/selection诊断；记录完整H与对角H对同一theta的预测差异，仅作归因，不产生第二候选。
4. 根据真实hard结果修订下一阶段算法设计或生成首个正式代表；合法可达且未全部回退者按活动计划做目标侧shard0/六shard。纯降时逐位要求不覆盖本精度候选。

旧A-GR2的STE/有限差分证据不是本卡启动门：本卡不使用其STE或M=I+N。硬round的无穷小差分多数为零，不能把它当真实平滑梯度；若补旧归因，必须区分平滑导数检验与有限步长硬码变化，且不为本卡额外开旧训练。
