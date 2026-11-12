# 精度分析（与主线 chat 并行，2026-11-12）

只读主线文件，所有输出在本目录。OOF 概率用 `runs/nn_fu_steps3000_s{0,1,2}`（v5/v6 的 TCN3000），模板只用训练折估计，口径同 `tools/template_decode.py`。

## 结论

1. **按日期判版本后，模板选择已不是瓶颈**：date 0.913 vs 已知正确模板（oracle）0.916（subject 折，s0）。剩下的误差都出在模板内部的对齐上。
2. **解码器两处免费修正**（推理耗时不变），三个种子平均：

| 解码设置（res 10） | subject | session |
|---|---|---|
| v6 现状（U 段填 argmax，结尾可免费截断） | ≈0.913 | ≈0.911 |
| + U 段按相邻段类别填充 | 0.9177 | 0.9175 |
| + 结尾锚定 cap=4 | **0.9226** | **0.9247** |
| 不用日期：现状 → 两处修正 | 0.885 → 0.897 | 0.897 → 0.912 |

   - U 段填充：U 秒原来输出 TCN argmax。运动后恢复期心率仍高，argmax 是 Activity，解码器把 Rest 划进结尾 U 时就会错判。改为只在前后相邻已计分段的类别里取 argmax。
   - 结尾锚定：原 `align()` 允许录音在任一段内提前结束、且短于均值不罚，丢掉了"从录音结尾倒推"这个锚点。各段起点从结尾倒推的 sd 远小于从开头算（AEROBIC V1 结尾 Rest 18 s 对 173 s；ANAEROBIC V1 13 s 对 82 s；STRESS V2 最后的 S 79 s 对 222 s）。现在短于均值也按高斯罚，最多罚 cap=4 nats，给真被截断的录音留余地。
   - 补丁：`decoder_fix.patch`（对 submission/adm_model.py 的 diff），完整文件 `adm_model_patched.py`。已验证与分析代码结果逐位一致。
3. **试过但没用的**：
   - res=5：+0.001，但解码从 5.5 ms 变成 15 ms，速度分亏更多。
   - 按协议分设 alpha：0.002/0.002 已是最优。
   - 时长 sd 缩放、shift：都更差。
   - S 段局部平移的二次精修：更差，因为 TCN 的 p(S) 峰值本身偏移 ±40 s。
   - 三种子集成、pos 特征、+transformer 集成：都没有增益。
4. **剩余瓶颈是 Stress F1 ≈ 0.845**：
   - 错误集中在短 S 段：V2 S3（Subtraction）81% 窗口错，V1 S2 和 V2 S2（opinion）37–40% 错。
   - TCN 确实有反应（V2 S3 段内 p(S)=0.60，段前 0.10），但时间上是模糊的。
   - 模型侧再提升要靠提高 S 起止的时间精度，不是靠调解码器。

## 脚本

- `err_decomp.py`：free / date / oracle 三种模板选择的对比，加上错误窗口按到真实切换点的距离分桶。
- `decode_variants.py`：U 段填充方式、alpha、res、shift、sd 网格。
- `stress_segments.py`：STRESS session 内按段统计错误率。
- `refine.py`：两阶段 S 段精修（负结果）。
- `align2.py`、`cens_test.py`：结尾锚定（cap）。
- `ens_test.py`：集成测试。
- `alpha_proto.py`：按协议分设 alpha。

## 第二轮：模型侧（2026-11-12）

`nn_cv_acc.py` 是 tools/nn_cv.py 的副本，新增两个开关：`--u_class 1` 把不计分秒作为第 4 类；`--bnd 1` 加一个边界头，预测 ±bnd_k 秒内是否有标签切换（BCE，pos_weight 20）。OOF 输出在 `oof/`，配套解码器是 `decode4.py`（U 段用 log p(U) 作发射，段起点加 beta × 边界对数几率）。

三个种子平均，res 10，cap 4，邻居填充：

| 配置 | subject | session | Stress F1 |
|---|---|---|---|
| 3 类（第一轮最优） | 0.9226 | 0.9247 | 0.843 |
| 4 类，U 发射 | 0.9270 | 0.9279 | 0.848 |
| **4 类 + 边界头，beta 0.05，U 发射** | **0.9340** | **0.9325** | **0.867** |

- 只训练成 4 类、解码时不用 U 发射：没有收益。
- beta 在 0.02–0.1 之间结果都稳定，0.2 开始下降。
- 调参（bnd_k 取 1 或 6，bnd_weight 取 3）：作业 21545003（1 张 V100 跑 6 个进程）。完成后作业 21549549 自动解码，结果写到 `logs/tune_summary.txt`。

## 第三轮（2026-11-12 下午）

| 配置（三种子平均） | subject | session | Stress F1 (subj) |
|---|---|---|---|
| u4b，beta 0.05，U 发射（第二轮最优） | 0.9340 | 0.9325 | 0.867 |
| **u4so（`--bnd 3`：通用 + Stress 开始 + Stress 结束三个头），beta 0.05 + beta_so 0.1** | **0.9365** | **0.9347** | **0.871** |
| u4so 只用通用头（beta 0.05） | 0.9330 | 0.9328 | — |
| 协议分类器（整段心率特征判有氧/无氧）作模板先验 | 0.9343 | 0.9326 | 负结果 |
| 另一个 chat 的 ph05（阶段辅助头，用 decode4 解码） | 0.9323 | 0.9317 | 没有增益 |

- 开始/结束头：subject 折三个种子都更好，session 折 3 个里 2 个更好。两折平均都是 +0.002 左右，约 +0.2 分。
- beta_so 的取值：session 折上 0.05–0.2 结果都差不多；subject 折上 0.2 会掉到 0.928。所以取 0.1。
- 协议分类器只分对 76–80% 的运动 session。最难的 ANAEROBIC V2（f07/f08/f11/f13）录音长，而且运动后恢复期像在继续有氧运动。
- 解码：`decode4.decode4(prob, T, ver, 4, 3, beta=0.05, u_emit=True, beta_so=0.1)`，prob 共 7 列 = 4 类 + 3 个边界头。
- 作业脚本：`arr.sbatch`（普通 V100 队列，数组）；`dbg_pack.sbatch`（gpu-debug 打包，已弃用：和另一个 chat 抢唯一的槽位）。

## v7 提交包（2026-11-12 15:15）

`runs/v7_u4so/submission_v7_u4so.zip`
- 模型：u4so，全量数据、单种子、3000 步。训练脚本是 `train_full_u4so.py`，作业是 `train_v7.sbatch`。
- 代码：`v7_src/adm_model.py`，由 12:55 的 submission/adm_model.py 加 `decode_u4so` 而来。打包脚本是 `package_v7.py`。
- 核对结果：
  - `decode_u4so` 与 decode4 在 OOF 上逐秒一致。
  - 现场特征与训练特征差 0。
  - ONNX 与 torch 差 1e-6。
- 单核实测（同一作业对比）：v7 32.2 ms，v6 31.3 ms。速度分 17.85 对 17.91。

## 速度轮与 v8（2026-11-12 16:30）

`runs/v8_p10/submission_v8_p10.zip`
- 模型：pool=10、空洞率 1,2,4,8，46k 参数。TCN 在 10 s 块上运行，平均池化和重复回 1 Hz 都在 ONNX 内部完成。
- 代码：`v8_src/adm_model.py`，即 v7 换成快版 `align4`（`align_fast.py`）。快版在 190 万秒上与 v7 逐秒一致，速度 1.59 倍。

| | CV subject/session | 单核平均/最慢（同一作业） |
|---|---|---|
| v7 | 0.9365 / 0.9347 | 24.7 / 40.6 ms |
| v8 | 0.9342 / 0.9352 | 20.9 / 34.7 ms |

- 负结果：
  - pool=5（0.9338/0.9316）。
  - 时长搜索窗口缩到 ±3 sd：精度掉，时间只省约 10%。
  - 两阶段选模板：没变快。
- 注意：`sbatch --export` 遇到逗号会截断，参数改用 `RUNFILE`（`arr.sbatch` 按行读取）。
