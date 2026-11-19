# 四档逐请求采集结果

run：`mry-maxpcp-main-1119`。实际发送/排空全程为2026-11-19北京时间 **02:02:06.773–02:37:58.162**。
仿真 service `31d907b1`，真实 Prefill `6842e3d9`。本采集任务未打流、改部署或调整模型参数。

总发送23,341条，客户端成功23,340条；按成功回执ID集合核对，**23,340条成功请求全部配对**。C512另有1条客户端timeout，未取得HTTP状态、业务request ID、首内容或usage；不能声称全部发送请求成功，也不能把这条计作成功真实请求漏复制。

## 引擎结果

口径：同请求、请求等权；Scheduler接纳→首次结果被Scheduler消费。差值为仿真减真实，单位ms。不是部署监控首包。

| 并发 | 成功/配对 | 真实TTFT | 仿真TTFT | 相对误差 | 首次dispatch前差 | 末dispatch→ready差 |
|---|---:|---:|---:|---:|---:|---:|
| 64 | 5,632/5,632 | 2290.639 | 1887.815 | −17.5857% | −398.828 | −3.928 |
| 128 | 5,741/5,741 | 5557.745 | 5240.828 | −5.70225% | −322.338 | +4.989 |
| 256 | 5,876/5,876 | 12017.026 | 12020.905 | +0.03228% | −1.872 | +5.300 |
| 512 | 6,091/6,091 | 24789.352 | 23962.879 | −3.33398% | −834.962 | +7.693 |

C64尚未达到引擎TTFT 10%目标；其他三档的引擎均值在10%内。监控吞吐/命中率及部署首包由主代理独立汇总，不能以本表代替。

## 输入与缓存

23,340对输入token全部相同。C64/C128/C256的逐请求命中及调度token全部相同。C512有4条不同，分别是同family两对 `000326↔000358`、`000602↔000634`：一侧先计算的请求在另一侧命中3072 tokens，另一请求则交换角色。两组相反差异抵消，故四档配对总hit tokens与总input tokens均完全一致；不能把总量一致写成C512逐请求全部一致。

差异请求保存在 [cache-different-requests.json](main-engine-summary/cache-different-requests.json)，逐请求命中一致性计数在 [cache-pair-summary.json](main-engine-summary/cache-pair-summary.json)。这里描述观察到的调度/缓存次序差异，不据此断言缓存实现有bug。

客户端timeout为 `mry-maxpcp-main-1119-c512-000314`，发送于02:32:36.284，处于10秒初始并发爬升。该条同时位于real/sim未配集合，成功回执集合无缺失。无业务ID，当前不能证明请求是否到过推理引擎。

## 证据

- 最终 [engine-summary.json](main-engine-summary/engine-summary.json)、[engine-summary.csv](main-engine-summary/engine-summary.csv)；[cases.json](cases.json)来自打流机run输出。
- 各case本地明细在 `../qwen38-tpu-pp-rate-20261117/maxpcp-main-1119-c{64,128,256,512}/analysis/`，包含paired-summary、coverage及逐请求CSV。
- 37原始trace：`/data2/maruiyan.mry/qwen38-tpu-pp/work/pressure-1116/maxpcp-main-1119-c{64,128,256,512}/traces/`。
- C256在事件导出完成后读取process元数据触发429，只补元数据后分析；C512读取ASI短暂transport failure，保留18 MiB已导出数据，在已完成的`events.000248.jsonl`末序号5110204之后，从000249继续。两者都是证据读取错误，未重发请求。
- C512 `--require-complete`按6092总回执返回非零；最终另按成功回执ID集合校验6091成功请求全覆盖，保留失败记录。统计工具有等总数但漏成功ID的反例校验，不只比较数量。

监控固定窗口（北京时间、右端不含）：C64 **02:04–02:07**，C128 **02:14–02:17**，C256 **02:24–02:27**，C512 **02:34–02:37**。
