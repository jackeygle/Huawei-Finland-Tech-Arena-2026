# Huawei Finland Tech Arena 2026 — Stress or Sweat?

任务分析（2026-11-10）。提交截止 **2026-12-01 23:59（赫尔辛基时间）**，每天最多 3 次提交。

## 目录

| 路径 | 内容 |
|---|---|
| `raw_page/` | 比赛网页全部子页面（html + 纯文本），含规则条款 `agreements.txt` |
| `downloads/` | 官方 Google Drive 的全部 94 个文件：85 个 parquet + 文档 + `adm_base_class.py` + `submission_guide.pdf` |
| `analysis/` | `eda_stats.py`（逐 session 统计）、`quick_baseline.py`（难度探测）、`plot_sessions.py`、`figs/`、`session_stats.csv` |
| `env313/` | 与评测环境一致的 Python 3.13 环境（numpy 2.5.4、pandas 3.1.1、sklearn 1.9.1、onnxruntime 1.31.1） |
| `.venv/` | 下载和探索用的 Python 3.9 环境 |

## 1. 任务

只用腕带的生理信号（没有加速度计），把每个时刻分成三类：`Rest`、`Stress`、`Activity`。

- 输入：一个 DatetimeIndex 的 DataFrame，列为 `BVP`(64 Hz)、`EDA`(4 Hz)、`Temperature`(4 Hz)、`HR`(1 Hz)、`IBI_dt`、`IBI_duration`（逐拍，不规则）。所有列放在 64 Hz 的时间轴上，低采样率的列大部分是 NaN。
- 输出：同索引的 DataFrame，一列 `Activity`。
- 提交：一个 zip（≤50 MB），里面是 `xxx.py`（暴露 `get_model()`）和同名 `xxx.onnx`。模型必须是 ONNX，特征工程可以写在 Python 里。
- 运行限制：1 个 CPU 核、禁多线程、内存 ≤4 GB。

## 2. 评分

总分 = 质量分（0–80）+ 速度分（0–20）。

- 质量分 = macro-F1 × 80。按 10 秒窗口比较，每个窗口取出现最多的类。三类 F1 等权，所以样本最少的 Stress 和另外两类一样重要。
- 速度分 = (1 − MET / 0.3 s) × 20，MET 是 `run()` 跑 50 次的平均耗时。

两者的换算：**每 15 ms 耗时 = 1 分 = 1.25 个百分点的 macro-F1**。一个 150 ms 的特征流程要靠多 12.5 个点的 macro-F1 才能回本，所以速度不是次要项。

同分时，评委按"是否适合低资源可穿戴设备"决胜，这也偏向小模型。

## 3. 规则里要注意的

- **禁止外部数据**，只能用官方给的数据训练和验证。
- 预训练模型可以用，但必须声明。AI 编程助手和开源库允许。
- 入围后要交全部源码和训练脚本，榜上分数必须能复现，否则可能取消资格。训练要固定随机种子、留好脚本。
- 禁止队伍之间共享代码或提交，禁止试图获取或重构评测数据。
- 参赛资格：欧洲高校在读（本/硕/博）或毕业两年内，18 岁以上，非华为员工。
- 最多 10 支队伍进赫尔辛基决赛（12-11，半天），路费每人最多报 €500。

## 4. 数据

36 名受试者，85 个 session，共约 44 小时。`Sxx` 是第一批（18 人，全男，协议 V1），`fxx` 是第二批（18 人，协议 V2）。

| 协议 | session 数 | Rest | Stress | Activity | 不计分 |
|---|---|---|---|---|---|
| STRESS | 31 | 774 min | 191 min | 0 | 88 min |
| AEROBIC | 27 | 100 min | 0 | 796 min | 27 min |
| ANAEROBIC | 27 | 117 min | 0 | 477 min | 68 min |
| 合计 | 85 | 991 min | 191 min | 1273 min | 183 min |

标签映射（`phases_to_activity.txt`）：Stroop / TMCT / Real opinion / Opposite opinion / Subtraction → Stress；Warm-up / 各 rpm 档 / Sprint / Cooldown → Activity；Baseline / Rest → Rest；`Undefined` 不计分。

观察到的事实：

- **Stress 只占计分时间的 7.8%**，而且每段很短。V2 的 Real/Opposite opinion 每段只有约 30 秒，即 3 个评分窗口。
- **Stress 和 Activity 从不出现在同一个 session**。Stress 只在 STRESS 协议里，Activity 只在两个运动协议里。
- **V1 和 V2 协议结构不同**。V1 的运动 session 开头没有 Baseline，Rest 只有结尾 2 分钟；V2 开头有 4–6 分钟 Baseline。V2 的 STRESS session 长一倍（约 45 min 对 23 min），多出来的都是 Rest。
- **Cooldown 算 Activity**。无氧协议里 Cooldown 占了 Activity 的大部分，此时人已经不在冲刺。
- **绝对值跨人差异很大**。EDA 中位数从 0.15 到 48 µS，所以按 session 归一化很重要（见第 5 节）。
- **心率在不计分段也会升高**。图里 STRESS 协议的 `Undefined` 段（任务说明/准备）HR 常常比 Stress 段还高。

数据质量问题：

- 3 个 session 的温度传感器故障，读数约 209 °C：AEROBIC/S07（全程）、ANAEROBIC/S02（92%）、ANAEROBIC/S05（62%）。
- 12 个 session 有超过 1 秒的时间断档，最长 270 秒（AEROBIC/f02、ANAEROBIC/f04）。
- IBI 在运动时大量缺失（AEROBIC/f04 整个 session 一个 IBI 都没有），所以不能依赖 IBI/HRV 特征，HR 列更稳。
- AEROBIC/f07 的阶段顺序异常（Cooldown 在 90-95 rpm 之前）。
- 提前结束的：AEROBIC S03、S07，ANAEROBIC S06。

图：`analysis/figs/session_S05.png`（V1）、`analysis/figs/session_f03.png`（V2）。

### 推断：评测集很可能是被拿掉的那些 session

`subject-info.csv` 里标了"做过"、但数据里没有的 session 有 12 个：

- STRESS：S02、S04、f07、f08、f14
- AEROBIC：S11、S13、f06
- ANAEROBIC：S01、S15、S16、f10

这是推断，官方没有说明。如果成立，有两个后果：

1. 评测集的受试者大多在训练集里出现过（别的协议）。验证时除了"按人分组"的交叉验证，还应该看"留出某人的某个协议"的情形。
2. 这份数据的结构和 PhysioNet 上公开的一个 Empatica E4 数据集（诱发压力 + 结构化运动）一致。**不要去下载那个公开数据集**：规则明确禁止使用评测数据的任何副本或来源，也禁止外部数据。

## 5. 难度探测

`analysis/quick_baseline.py`：每秒一组特征（BVP 幅度、EDA、温度、HR、IBI，加 10/30/60/120 秒的居中滚动均值、标准差和斜率），梯度提升树，按受试者分 6 折交叉验证，按官方方式在 10 秒窗口上算 macro-F1。没有调参。

| 设定 | macro-F1 | 质量分 | Rest F1 | Stress F1 | Activity F1 |
|---|---|---|---|---|---|
| A：只用局部窗口特征 | 0.581 | 46.4 / 80 | 0.734 | 0.198 | 0.809 |
| B：A + 减去本 session 中位数 | 0.730 | 58.4 / 80 | 0.788 | 0.510 | 0.893 |

- **Stress 是瓶颈**。设定 B 里 1201 个 Stress 窗口只认出 558 个，568 个被判成 Rest。
- **session 级归一化把 Stress F1 从 0.20 提到 0.51**。这依赖 `run()` 收到的是整段 session。
- 这两个数是没调参的探测结果，也没算速度分，只说明量级。

## 6. 还不知道的事

1. **`run()` 一次收到多长的数据**：整个 session，还是一小段？这决定能不能做 session 归一化，也决定 0.3 秒够不够用。一个 30 分钟的 session 有 11.5 万行。
2. **评测集是否只含上面推断的那 12 个 session**，以及是否和训练集格式完全一致（例如 `Phase`、`user_id` 列肯定没有）。
3. **输出要不要每个时间戳一行**。官方示例是 `index=input.index`，按这个做。

第 1 条可以用一次提交探出来：交一个极简模型，从得分和耗时反推。

## 7. 建议的路线

1. **先交一个能跑通的极简提交**，确认格式、拿到速度分的基准，并探明第 6 节的问题 1。
2. **搭好本地评测**：按官方口径算分（10 秒窗口 macro-F1 + 在 `env313` 里单核计时 50 次），两种交叉验证都跑。
3. **特征用 numpy 写，不用 pandas rolling**。先降到 1 Hz 或 4 Hz 再算，目标是整个 `run()` 在 30 ms 以内（只丢 2 分）。
4. **模型选小的**：梯度提升树或小型 1D-CNN/GRU，导出 ONNX。树模型在单核上通常是几毫秒。
5. **重点攻 Stress**：session 内相对特征、EDA 的相位成分（SCR 峰）、HR 相对基线的变化、预测后的时间平滑（Stress 段至少持续 30 秒）。
6. **决赛答辩材料**：Drive 里说有 proposal template，但文件夹里没有这个文件，需要在平台上确认。


## GitHub 完整项目快照

本仓库保存当前项目中可提交的文件，包括源码、实验说明、官方数据及文档、训练与评估结果、图表、日志、模型权重、ONNX 文件、提交 ZIP，以及 `.venv/`、`env313/` 和 `.pipcache/` 的本地环境与缓存快照。

`external/` 中的第三方源码、原许可证和现有模型文件也一并保存。根目录和嵌套目录的 `.git` 元数据，以及认证凭据不纳入项目文件快照。`downloads/LICENSE.txt` 为官方数据集许可证；第三方依赖各自的许可证随文件保留。

符号链接按链接本身保存，不复制项目目录以外的目标内容。环境、脚本、Slurm 参数及部分符号链接保留原服务器路径，换机器后需按目标系统重新配置；已提交的 Python 环境不保证可以直接迁移运行。
