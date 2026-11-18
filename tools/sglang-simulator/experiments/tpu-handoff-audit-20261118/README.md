# 当前时间模型与四档 PP 交接核对

2026-11-18；分析2026-11-17晚间64/128/256/512并发数据。仅单副本Prefill，KVCM/tracker关闭。本次没有更换参数、部署或重新打流。

结论：原生调度/流水线控制被复用，不等于所有执行成本和32-rank同步都被原样复用。forward有效窗口的均值预测较准；barrier停留存在压力相关差异；ICI纯传输完成耗时仍未单独观测，因此当前0.8ms只能称配置值，不能宣称已验证其传输误差。

## 复用与替代的完整清单

| 时间环节 | 当前实现与生效参数 | 精度边界 |
|---|---|---|
| 批次/PCP特征 | 自行实现StageSpan/PcpLayout、连续token流interleave=128、PCP8和padding桶映射，供22/26维模型使用 | 是按原生布局规则实现的特征转换，并非直接调用原生布局函数；没有独立加时，但错误会改变预测输入 |
| P0/P1/P2执行 | 各自22维stage ML；按PCP padding附加残差。8192档P0/P1/P2为+1/+3/+6ms | 标签是forward入口到local drain ready，包含stage内部计算、PCP/EP通信与主机观测影响；不是纯kernel时间 |
| P3执行与输出 | 26维stage ML，含prompt完成数等尾部特征；8192档残差0ms | 标签到output ready observed，包含原生末stage尾部、采样/拷贝；没有再额外叠加通用GPU采样预测器 |
| 各stage输入准备 | `max(0,3.773523+4.896685*N−1.809220*Npast0)` ms，在before_step之后、start_forward之前执行；各stage同一公式 | Npast0是past KV长度为0的成员数，不等于Scheduler的新请求列表长度；四档目标均值约22.41–22.51ms。本轮未逐stage重验该标签 |
| Scheduler元数据构建 | 在原生`_build_kv_connector_meta`处，目标为`2.067936+2.220581*Ncached+0.001287248*Q` ms；只补目标减实际已运行的正差 | Q是本批scheduled token数；Ncached是已在running列表且本步有token的请求数，不是前缀命中数。四档目标均值约14.03–14.07ms；不是整个查询/首次调度等待的模型 |
| WORLD barrier | 原生PPWave交接时序保留，硬件实现换成4个CPU stage进程的Barrier；会合后每rank并行补1.3ms | 真实是32个逻辑worker，仿真没有逐一展开PCP8的到达抖动。1.3ms来自此前代表rank剩余等待校准，不是整个barrier停留时间 |
| PP激活数据 | 环形Pipe交换step身份和ready_at；收到描述符后将激活就绪时间设为当前时间+0.8ms | 不传真实张量，不使用字节/带宽模型；不依并发变化。0.8ms可与主机准备部分重叠，不能直接替代同步阻塞 |
| RPC、回复与结果消费 | 虚拟executor用本机Pipe和CPU worker替代真实进程/机器传输；复用原生FutureWrapper、FIFO drain和LazyPushStepper消费顺序 | 未单独校准RPC/序列化/唤醒延迟；仍有当前CPU的实际运行耗时。四个模拟stage也不是32个真实worker的逐一重放 |
| CPU runner与控制开销 | 内层UniProcExecutor同步生成模拟输出/connector簿记，特征构造、Python控制循环实际运行；TPU模式跳过通用整步GPU预测 | 当前没有把所有这些CPU成本映射成真实服务CPU成本；保留逻辑不代表耗时相同。不能把`ignore_cpu_overhead`理解为运行时所有CPU工作都不耗时 |
| V6D数据面与完成信号 | 保留原生查询、pending/recheck、分配、job/rank确认与延迟释放；TPU worker的load/store DMA被完成集合和job结果替代，fetch只走完成/封存协议 | **当前TPU路径没有DMA带宽时间模型**；仓库通用GPU HybridConnector的BandwidthModel不能视为这里已激活。DRAM命中加载、写回及副本内分片取数成本尚未验收 |
| Lazy辅助状态 | 初始measured seed=78.861ms；空step携带connector状态的队列语义由adapter补齐 | seed影响早期push估计，不是每请求固定加78.861ms。`empty_step_ms`未配置；旧`PREFILL_HANDOFF_MS=0`，不额外加后置等待 |
| 范围外链路 | KVCM/tracker关闭；本地Raiden替代P→D生命周期；网关/Chat/部署入口仍实际运行 | 不预测跨域网络、Decode及真实P→D传输；不能把外层监控差值塞进PP stage模型 |

原生复用的核心是Scheduler的选请求/预算/分块/KV状态与释放逻辑，以及LazyPushStepper的槽位、push/settle、FIFO/Future消费和PPWave的before/start/end_forward次序。硬件构造、同步、传输、耗时与上述适配点由仿真替换；因此“只替换forward，其他完全原样”不准确。

## 四档forward有效窗口

同一真实批次输入当前部署预测器，单位ms，单元格为实测/预测（均值相对偏差）。这是同批评估，6976批×4stage；不是纯设备kernel精度。

| 并发 | P0 | P1 | P2 | P3 |
|---|---:|---:|---:|---:|
| 64 | 145.03 / 146.23 (+0.83%) | 145.48 / 144.05 (-0.98%) | 147.17 / 147.18 (+0.01%) | 141.05 / 141.43 (+0.27%) |
| 128 | 145.27 / 146.21 (+0.65%) | 145.68 / 144.11 (-1.07%) | 147.35 / 147.17 (-0.12%) | 141.11 / 141.31 (+0.15%) |
| 256 | 145.30 / 146.21 (+0.62%) | 145.72 / 144.08 (-1.13%) | 147.45 / 147.20 (-0.16%) | 141.38 / 141.37 (-0.00%) |
| 512 | 145.18 / 146.22 (+0.72%) | 145.60 / 144.16 (-0.99%) | 147.33 / 147.31 (-0.01%) | 141.16 / 141.36 (+0.14%) |

## 四档barrier停留

区间为同进程的barrier enter→exit。仿真包含实际四进程会合和补充的1.3ms；真实包含等其余world worker和原生barrier开销。它不是纯通信耗时，四stage并行等待不能相加。

取每档相同压力窗内，四个stage均完整的launch组；真实每stage为PCP0，仿真为其逻辑stage。同一侧通过protocol session/burst/ordinal配齐，**两侧不是同批成员配对**，不能将均值差当成逐交接预测MAE。真实/仿真请求投递仍有缺失。

单元格为真实/仿真均值ms。

| 并发 | 完整launch组真实/仿真 | P0 | P1 | P2 | P3 |
|---|---:|---:|---:|---:|---:|
| 64 | 1715 / 1694 | 5.03 / 6.29 | 29.82 / 32.40 | 28.29 / 31.08 | 13.91 / 14.30 |
| 128 | 1730 / 1719 | 4.68 / 6.17 | 30.09 / 32.53 | 28.49 / 31.53 | 13.46 / 14.76 |
| 256 | 1751 / 1731 | 6.13 / 6.11 | 31.76 / 32.28 | 30.16 / 31.41 | 14.28 / 14.59 |
| 512 | 1806 / 1788 | 7.63 / 6.04 | 33.83 / 32.12 | 32.29 / 31.43 | 16.78 / 14.68 |

另按每个launch取四个代表rank的最短barrier停留，再求均值。这只用各进程自身时长，不跨机器减时间；它不是最后一个真实worker到达后的纯barrier开销，因为其余28个worker未展开。

| 并发 | 最短停留：真实均值/中位数/P95 ms | 最短停留：仿真均值/中位数/P95 ms | 最短停留>100ms次数：真实/仿真 |
|---|---:|---:|---:|
| 64 | 1.211 / 0.577 / 4.343 | 1.415 / 1.396 / 1.441 | 0 / 0 |
| 128 | 1.068 / 0.583 / 3.496 | 1.418 / 1.396 / 1.436 | 0 / 0 |
| 256 | 1.810 / 0.580 / 4.315 | 1.410 / 1.396 / 1.435 | 1 / 0 |
| 512 | 3.797 / 0.589 / 6.542 | 1.413 / 1.396 / 1.439 | 5 / 0 |

真实中位数四档均约0.58ms，仿真约1.40ms；真实P95和极端尾部更宽。512档3.80ms均值不代表每次同步都需3.80ms，不能据此把1.3ms常数整体调大。下面保留所有>100ms的共同停顿，均发生在运行过程中，不是开头prologue；不从主表删除。

| 并发 | 北京时间（P0 barrier返回附近） | P0当轮交出的step | burst/ordinal | 四个PCP0最短barrier停留ms |
|---|---|---:|---|---:|
| 256 | 11-17 22:00:12.714 | 287268 | 1524/1395 | 1061.545 |
| 512 | 11-17 22:09:34.299 | 289216 | 1525/1097 | 830.673 |
| 512 | 11-17 22:10:05.885 | 289388 | 1525/1269 | 815.889 |
| 512 | 11-17 22:10:24.281 | 289485 | 1525/1366 | 1112.096 |
| 512 | 11-17 22:10:33.003 | 289533 | 1525/1414 | 201.124 |
| 512 | 11-17 22:10:45.691 | 289598 | 1525/1479 | 1076.010 |

同一launch中不同stage携带不同批次，表中step只标识P0携带的批次。秒级停顿在四个被观测rank内均发生；候选原因包括其余PCP rank晚到、原生CPU barrier及进程调度异常，当前尚未区分。forward标签结束在后续WORLD barrier之前，因此forward误差表不会覆盖这些等待。该差异不是ICI纯传输时间的证据，也尚不能把整组TTFT差额全部归给这些事件。


## 提交窗口与传输不可辨识边界

以下是真实每stage的barrier返回→handoff提交返回均值ms，包含主机侧调用/提交开销，**不是提交→传输完成**。

| 并发 | P0 | P1 | P2 | P3 |
|---|---:|---:|---:|---:|
| 64 | 0.748 | 0.650 | 0.665 | 0.322 |
| 128 | 0.751 | 0.646 | 0.666 | 0.323 |
| 256 | 0.759 | 0.645 | 0.672 | 0.330 |
| 512 | 0.753 | 0.648 | 0.670 | 0.331 |

本次真实PCP0每次交接报告的payload字节数集合：`[16777216]`。这是每rank的逻辑张量量，不是已测线速或wire字节数。


原生`_launch_now`在op前后调用`sync.synchronize(..., wait=False)`；采集的`handoff_submitted`明确为submission_only。`start_forward`返回后记录`receive_reference_returned(receive_device_ready=False)`，所以无法从这两个host时间戳求独立ICI传输完成耗时。

当前forward标签从主机forward入口开始，未独立扣除可能仍在途的输入传输依赖。因此显式0.8ms与有效forward窗口之间**存在边界重叠的可能性，尚未完成排重验证**。不是已经证明重复计时0.8ms，也不能直接删掉该常数。需要设备/依赖完成证据或受控实验才能区分。

真实barrier返回后的提交窗口与仿真Pipe开销+0.8ms激活就绪是不同端点；不把二者强行计算传输误差。四档纯ICI完成时间及其预测误差当前均为未验证。

## 数据完整性和下一步

真实各进程用各自单调时钟相减；wall/mono anchor只用于选压力窗，不跨机器减时间。四个PCP0不是32个rank的全量同步证据。launch级明细保留prologue、push/settle与长尾；单侧不完整组/端点在summary中显式保留。

优先按上述六个launch检查其余PCP rank，区分谁晚到与barrier本身阻塞；同时把host preparation、host提交、输入依赖和forward标签的边界排重。再通过固定同批的交接回放检查等待因果。保持同一组成员和成本再做敏感性实验，避免把分批/到达/漏复制差异拟合成通信常数。尚不据本表修改0.8/1.3ms或按并发加惩罚项。

## 版本、代码和证据

- 本轮服务未变化：真实collector92a8db0/revision10；仿真wheel3788f74/revision19、P0模型41a7705、P3 r2模型7ce9ff0。源码核对个人分支228c23d与3788f74在simulator/src下无差异。
- 源码根（37）：`/data2/maruiyan.mry/qwen38-tpu-pp/work/pp-lifecycle-20261112/real-barrier-1113/tools/sglang-simulator/src/sglang_simulator/`。
- `simulation/vllm/tpu_pp/stage_runtime.py:118`交接adapter；`:154`stage执行；`:163`padding残差；`:379`RPC/Future；`:589`Lazy适配。
- `simulation/vllm/tpu_pp/stage_model.py:156`host准备；`scheduler_timing.py:9`metadata目标补差；`native_control.py`原生控制抽取。
- `simulation/vllm/v6d/tpu_connector.py:282,381`TPU完成信号/job协议；`v6d_manager.py:99,162`数据fetch替代。
- `collector/vllm_hook/native_boundaries.py`原生barrier enter/exit、submission_only、receive_device_ready=False；`simulation/vllm/tpu_pp/vendor/pp_wave.py`原生异步提交与同步顺序。
- 37原始新增导出与分析：`/data2/maruiyan.mry/qwen38-tpu-pp/work/pressure-1116/ramp-handoffs-1118/`；原生forward明细沿用`ramp-predictor-1117/`。
- 原项目实验目录`experiments/2026-11-18-time-model-audit/`：`handoff-summary.json`、`handoff-observations.csv`、`case-windows.json`、采集来源manifest与`analyze_handoffs.py`。完整原始事件留在37。
