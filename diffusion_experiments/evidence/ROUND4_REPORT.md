# Round4 Report — Joint Semantic–Preference Diffusion with Locked Test Evaluation

日期：2026-10-06。协议：`ROUND4_JOINT_SEMANTIC_PREFERENCE_V1`。数据集：Baby。

本轮严格先完成 stage0、smoke、4 个 formal fit、DEV 两 seed 共同选择与 INTERNAL 评价；随后依据用户本轮明确授权，在 selection lock 固定后一次性打开 Test。Test 没有参与 checkpoint、eta、L、seed 或特征选择，Test 后未做任何参数修改。

## 结论先行

1. **是否避开旧 energy / one-hot 路线？——是。** 本轮直接使用共享表示上的 64D T/V Gaussian diffusion noise prediction 与 MASK 偏好解码，最终依据 unary preference logit 重排，不使用 Round2/3 residual 或重建 energy 差，也不生成 TopL one-hot。
2. **自然训练分布是否更匹配？——是，但边界有效监督没有随 TopL 同比例增加。** L100 有 2,170 个 matched TRAIN query / 808 个 A 内 matched query；L500 为 4,525 / 832。degree 和 CF-norm 捷径诊断均约 0.46–0.49，没有触发 shortcut gate。Top500 把自然 probe 支持从 21.44% 提到 43.37%，但 A 内正例仅从 812 到 836。
3. **扩大 TopL 本身带来多少变化？——开发集小幅正、Test 略负。** DEV 上 B500 相对冻结 B100 的 U 为 **+0.7025%**；同口径 strict-FIT Test 上 B500 相对 B100 为 **−0.1495%**。因此不能把扩大候选池本身当成稳定收益。
4. **Diffusion 单独增加多少？** L100：DEV **+0.3725%**、INTERNAL **+0.0840%**、TEST **+0.1567%**；L500：DEV **+0.3794%**、INTERNAL **−0.9820%**、TEST **+0.4192%**。
5. **两 seed 和开发集合是否一致？** L100 两 seed 在 DEV、INTERNAL、Test 都为正，但幅度很小。L500 两 seed DEV/Test 都为正、INTERNAL 两 seed都为负，明显不稳定。
6. **达到 1% 吗？——没有。** 没有任何 L 同时达到 DEV/INTERNAL 1% 开发门槛；Test 的平均增量也只有 L100 +0.1567%、L500 +0.4192%。
7. **最终分类。** L100：`POSITIVE_BELOW_TARGET`；L500：`VALID_NO_INCREMENT` / split-unstable。Test 的正结果不能覆盖 L500 已冻结的 INTERNAL 负证据。
8. **Test 对比口径。** Round4 主表使用 strict-FIT 同一 backbone 的 MSCA → MSCA+CoLiftRec B_L → B_L+Diffusion。历史 full-training canonical seed1000 的 MSCA/CoLiftRec Test 只单列作上下文，不能作为 Round4 增量分母。

## 1. Stage0：候选深度、自然监督与基线变化

| L | TRAIN probe自然覆盖 | matched queries | A内matched queries | shortcut | DEV B_L vs B100 U |
|---:|---:|---:|---:|---|---:|
| 100 | 21.44% | 2,170 | 808 | PASS | 0.0000% |
| 200 | 29.63% | 3,051 | 812 | PASS | +0.5314% |
| 353 (5%) | 38.11% | 3,957 | 833 | PASS | +0.6080% |
| 500 | 43.37% | 4,525 | 832 | PASS | +0.7025% |
| 705 (10%) | 48.91% | 5,122 | 820 | PASS | +0.7236% |
| 1000 | 55.35% | 5,808 | 814 | PASS | +0.7340% |

正式训练只注册 L=100 / 500，没有因为 DEV 基线结果追加其他深度。L100/L500 的 degree shortcut paired win 分别约 0.470/0.465，CF-norm shortcut 约 0.483/0.471；边界方向也没有反转，因此 `MATCHING_SHORTCUT_REMAINS` 未触发。

## 2. 实现与正式训练

- 共享模型参数量：57,585。
- 连续状态：Round3 已校验 T32 + V32 = 64D；CF 不作重建对象。
- user context：129D，来自 strict FIT collab user、FIT-history T/V 均值和 log history length。
- candidate condition：13D；标准化只拟合 reranker TRAIN 自然候选分布。
- 50-step cosine Gaussian forward；标签 `{0,1,MASK}`，独立 `t_x/t_y`；部署固定 clean semantic + fully MASK preference。
- 前 100 updates 仅 semantic diffusion warmup；总 1,200 updates；checkpoint 固定 200/400/800/1200。
- formal seeds：202610065 / 202610066。
- smoke 和 formal 中 diff / pref / deploy 对 shared encoder 的梯度均为有限非零。
- 4 个 formal run 均绑定代码 commit `3a4d36d1763b07188f8e29a0d5fd70a8ea914263`，训练阶段均未读取 DEV / INTERNAL / CONFIRM / Test。

## 3. DEV 锁定与 INTERNAL

DEV 两 seed 平均、同一 L 统一选择：

| L | locked update | eta | mean DEV U | seed65 DEV U | seed66 DEV U |
|---:|---:|---:|---:|---:|---:|
| 100 | 400 | 0.05 | **+0.3725%** | +0.4255% | +0.3194% |
| 500 | 200 | 0.20 | **+0.3794%** | +0.2996% | +0.4593% |

Selection lock SHA256：`6e3a9efc620e63b84710a33ac9236ea00870271abdb2afed4c75c56395992f97`。

锁定后才评价 INTERNAL：

| L | seed65 INTERNAL U | seed66 INTERNAL U | mean INTERNAL U | 分类 |
|---:|---:|---:|---:|---|
| 100 | +0.1434% | +0.0245% | **+0.0840%** | POSITIVE_BELOW_TARGET |
| 500 | −1.3777% | −0.5862% | **−0.9820%** | VALID_NO_INCREMENT |

L100 的自然 INTERNAL pair win 约 0.502/0.512，A 内约 0.498/0.521，说明即使整体指标有小正增量，候选偏好判别仍接近随机。L500 INTERNAL 明显回撤，因此不允许根据后续 Test 重新选 update/eta。

## 4. 锁后 Test：strict-FIT 同口径主对比

Test evaluator 在打开标签前先做 DEV dry-run：L100/L500 的自然 MSCA candidate set 和 B_L candidate set 均为 **0 mismatch**，锁定后的 DEV mean U 均逐位复现。Test evaluator commit `3e20825adbe42fc620f4eda6650ba4e6b5aca01b` 在 Test 运行前已 push。Test 共评价 19,445 个用户；`TEST_USED_FOR_SELECTION=false`。

| strict-FIT Test | R10 | N10 | R20 | N20 | R50 | N50 | 主要相对变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MSCA | 0.060359 | 0.033007 | 0.089854 | 0.040598 | 0.155427 | 0.053886 | anchor |
| MSCA + CoLiftRec B100 | 0.064112 | 0.034999 | 0.094269 | 0.042775 | 0.161624 | 0.056357 | vs MSCA **+5.6324% U** |
| B100 + Round4 Diffusion（2-seed metric mean） | 0.064240 | 0.035050 | 0.094436 | 0.042819 | 0.161624 | 0.056366 | vs B100 **+0.1567% U** |
| MSCA + CoLiftRec B500 | 0.063552 | 0.034835 | 0.094732 | 0.042883 | 0.161382 | 0.056343 | vs MSCA **+5.4717% U**；vs B100 **−0.1495%** |
| B500 + Round4 Diffusion（2-seed metric mean） | 0.063810 | 0.034921 | 0.095353 | 0.043041 | 0.161382 | 0.056361 | vs B500 **+0.4192% U** |

这里的 B100/B500 是同一 strict-FIT backbone 下按各自候选深度定义的完整 CoLiftRec 基线。B500 在 Test 上相对 B100 略降，因此不能把 L500 的最终变化全算作扩大候选池收益。L500 两 seed 最终系统相对 strict B100 的 U 分别为 +0.1685% / +0.3706%，均值约 **+0.2695%**。

### 4.1 两个冻结 seed 的 Test 增量

| L | seed | Test U vs B_L | 95% paired-bootstrap CI | Top10 net | Top20 net |
|---:|---:|---:|---|---:|---:|
| 100 | 202610065 | +0.1675% | [−0.1127%, +0.4691%] | +3 | +3 |
| 100 | 202610066 | +0.1459% | [−0.1209%, +0.4466%] | +2 | +3 |
| 500 | 202610065 | +0.3191% | [−0.1258%, +0.7259%] | +4 | +6 |
| 500 | 202610066 | +0.5193% | [+0.0481%, +1.0150%] | +3 | +22 |

L100 两 seed 的 Test 增量均为正，但 bootstrap CI 都跨 0，不能称统计确定。L500 seed66 的条件化 paired-bootstrap CI 为正，而 seed65 跨 0；更重要的是 L500 在先前锁定的 INTERNAL 两 seed均为负，所以整体仍判定 split-unstable，而不是 Test 成功。

## 5. 历史 full-training canonical Test 对比（仅上下文）

| 历史 canonical seed1000 full-training Test | R10 | N10 | R20 | N20 | R50 | N50 |
|---|---:|---:|---:|---:|---:|---:|
| MSCA | 0.068212 | 0.037742 | 0.103716 | 0.046890 | 0.172154 | 0.060786 |
| MSCA + CoLiftRec | 0.071377 | 0.039091 | 0.107553 | 0.048376 | 0.175070 | 0.062070 |

历史 full-training CoLiftRec 相对 MSCA 的四主指标平均相对增益为 **+3.7713%**，4/4 primary、6/6 overall 为正。其绝对指标高于当前 strict-FIT 原型，因为训练图和协议不同；**不能**用历史 0.071377 等数值作为 Round4 Diffusion 的分母，也不能把 strict-FIT Diffusion 的 +0.1567%/+0.4192% 直接叠加到历史 full-training 指标。若将来要形成论文最终 Test 表，必须在完整 TRAIN 部署协议上重新冻结并评价 Round4 模块。

## 6. 科学判断

### L100：POSITIVE_BELOW_TARGET

- DEV mean U **+0.3725%**；INTERNAL **+0.0840%**；Test **+0.1567%**。
- 两个 seed 在三个集合都为正，方向是本轮最一致的配置。
- 但幅度远低于 1%，自然 INTERNAL pair 判别接近随机，Test bootstrap CI 两 seed都跨 0。
- 因此只能称“有小幅、跨 split 方向一致的开发/Test 增量”，不能称 Baby 目标完成，也不能作为 Diffusion 已稳定有效的强证据。

### L500：VALID_NO_INCREMENT / split-unstable

- DEV **+0.3794%**，Test **+0.4192%**，但 INTERNAL **−0.9820%**，两个 seed均负。
- B500 自身 DEV 比 B100 +0.7025%，Test 却 −0.1495%，候选扩展也存在 split 不稳定。
- Test 结果是在 INTERNAL 之后、参数冻结后得到，不能反过来重选或覆盖负的 INTERNAL 证据。

总体而言，本轮较 Round3 更好地避免了训练分布捷径和 energy 方向问题，并在 L100 获得了跨 DEV/INTERNAL/Test 的一致小正方向；但 **1% 目标仍未达到，且 L500 证明“更深候选池 + 联合扩散”并不稳定**。当前不应在看过 Test 后继续调 eta、checkpoint、L、窗口或 seed。

## 7. 实现、版本与曝光限制

- Round4 正式实现 commit：`3a4d36d1763b07188f8e29a0d5fd70a8ea914263`。
- 锁后 Test evaluator commit：`3e20825adbe42fc620f4eda6650ba4e6b5aca01b`。
- Selection lock SHA256：`6e3a9efc620e63b84710a33ac9236ea00870271abdb2afed4c75c56395992f97`。
- Test evaluator 打开标签前 dry-run：L100/L500 MSCA/B candidate set mismatch 均为 0；锁定 DEV U 精确复现。
- 当前 Test 共 19,445 用户；`CONFIRM_ACCESSED=false`，`TEST_ACCESSED=true`，`TEST_USED_FOR_SELECTION=false`。
- Baby Test 在本项目历史上已经多次暴露，因此即使本轮严格 post-lock 执行，也不能称 fresh untouched external confirmation。
- 当前 backbone 是 strict-FIT 原型，不是论文 full-TRAIN 部署模型。
- Test 后没有代码/参数重调；后续方向应交回 Advisor，不由 Executor 基于 Test 继续搜索。

交付文件：`ROUND4_REPORT.md`、`round4_protocol.json`、`round4_results.csv`、`round4_checks.json`。本地完整运行证据位于 `diffusion_experiments/runs/round4/`。
