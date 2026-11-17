# test4 共用 Chat 切换与验收（2026-11-17）

## 当前状态

已于北京时间17:59完成切流。新共享Chat 4/4 Running；真实/仿真分别直接路由到原TPU Prefill和原CPU1。16条和256条两次小批全部成功、全部逐请求配对，输入token及实际计算token全部一致。

18:17:59.229开始正式开放环速率实验`mry-shared-rate-main-1117`：18.5、19.2、20.5 RPS，各600秒，客户端排空后间隔300秒，最多20000条/组，inflight上限1024，单请求超时180秒。只按时间发送，达到上限/发送迟到则停止接收新请求，不偷偷降速。后台PID769337；当前尚无三组最终结果。CPU/TPU引擎与时间预测模型本轮均未更新。

## 当前拓扑

`POC → 共享Chat → 真实test4应用入口 → 原TPU Prefill`

在真实test4应用入口100%复制至`个人仿真应用入口 → 原CPU1 Prefill`。复制点位于唯一一次Chat预处理之后。

| 对象 | 标识 |
|---|---|
| 共享Chat应用 | test/464482ce：qwen3.8-max-hisim-pp-mry-test4-shared-chat，v-d57be948 |
| 共享Chat部署 | qwen3.8-max-hisim-pp-mry-test4-shared-chat-gateway，service 2e7b7a08，4实例 |
| POC模型 | pre-qwen3.8-max-hisim-pp-mry-test4-shared-chat |
| 真实应用 | qwen38-debug/bb1035d8：xiaohao-qwen3.8-max-L20G7-PP-test-4，v-a4cd4623 |
| 真实Prefill | -4b20，service 6842e3d9，应用权重1；旧Chat -5aa8权重0 |
| 仿真应用 | test/464482ce：qwen3.8-max-hisim-pp-mry-test4，v-e7e082b6 |
| 仿真Prefill | -cpu1，service f3ded9ec，应用权重1；旧-chat权重0 |

镜像比例1，当前到期时间2026-11-17 20:00。原两个Chat部署保留但不接实验流量。新模型已邀请既有授权Key ID4649685；密钥不写入文档或文件。

## 线上配置基准与改动

用户给出的Prefill部署是`prod/33e6d810/qwen3.8-max-2026-09-02-think-with-tools-prefill-eacc`；所属应用虚拟服务为`v-d88007e7`。关联Chat应用`qwen3.8-max-2026-09-02-mint`的`-4ab5`部署确实指向该Prefill应用。本次读取时该Chat有500实例，使用与test4相同的`dashllm_text:cpu_cp312_test_text_d9d22bec_20260902_004243`。线上同时存在其他4-engine HTTP/gRPC版本，本次未混用。

共享Chat沿用这套兼容基准：`DS_LLM_MULTI_ENGINE_NUM=8`、`runtime_config.model_instances=8`、`DashScope_Turbo_Batch_EngineNum=8`，以及超时、重试、缓存标记和tokenizer相关设置。实验使用4个CPU实例，资源池`cpu-mixed-8c16g-cn-beijing`。

必要适配：所有推理后端`*_SERVICE_ID`均指向test4**应用**`v-a4cd4623`；缓存相关模型白名单适配新POC别名；保留原test4的`ENABLE_REMOTE_CALL_CHAT=0`及已有Chat缓存连接配置。未引入生产推理后端，也未照搬生产副本数。脱敏配置见`shared-chat-target.sanitized.json`，完整私有快照仅存临时目录。

真实PD内部Decode服务引用保持`v-d2e2639a`，用于现有真实返回链路；仿真仍只处理Prefill，保留既有`DS_LLM_PD_EXTERNAL_ROUTER=0`和无Decode目标的配置。KVCM仍关闭，单机V6D路径不变。Chat增强缓存标记不是KVCM，不混为一谈。

打流机原`pp042/run.py`硬绑定旧应用/工作区，因此新建`sender-shared-chat.py`只调整精确目标绑定，新建`target-shared-chat.json`。请求生成、发压节奏和SSE计时逻辑不变；旧文件保留。Workbench重连会话555dgqbwuk，授权Key仅存在shell及其子进程环境。

## 小批验收

| 批次 | 北京时间 | 真实/仿真/配对 | token对齐 | 引擎TTFT真实/仿真 | 仿真相对真实 |
|---|---|---|---|---|---|
| shared-smoke-1117-r4 | 18:04:47.873–18:04:54.034 | 16/16/16 | 16/16 | 868.2/846.4 ms | −2.51% |
| shared-verify256-1117-r4 | 18:13:53.657–18:14:58.745 | 256/256/256 | 256/256 | 680.5/707.9 ms | +4.04% |

两批4RPS，全部HTTP200且SSE正常结束；输入token和scheduled query token逐条相同。256条最大发送偏差2.863ms。引擎TTFT是调度器接纳到消费首结果，不能替代部署监控首包，也不能证明更高压力下永不丢请求。

两侧原始trace及配对结果位于37：`/data2/maruiyan.mry/qwen38-tpu-pp/work/pressure-1116/{shared-smoke-1117-r4,shared-verify256-1117-r4}`；本地同名目录保存回执和analysis。

## 平台路径与外层开销

首请求619488d0-2818-92f7-8823-0c052921fd63：主请求进入共享Chat service2e7b7a08，镜像直接进入CPU1 servicef3ded9ec，`replicated_v2=true,forward=false,status=200`。此路径没有旧复制分支的perf→prod转发连接，但仍经过平台复制pod，不能仅据结构保证可靠性。证据：`shared-smoke-route-evidence.json`。

对256条按四种长度、五个时间位置抽样20条，20/20取得两侧有效Prefill平台记录（过滤backend_first_latency<0的诊断行）：

| 指标 | 真实 | 仿真 |
|---|---:|---:|
| 同样本引擎TTFT | 694.22 ms | 756.14 ms |
| 后端首包 | 716.75 ms | 776.85 ms |
| 部署首包 | 837.22 ms | 835.52 ms |
| 部署首包减后端首包 | 120.47 ms | 58.67 ms |
| batch_queue_latency | 105.65 ms | 10.30 ms |
| batch_enqueue_duration | 4.56 ms | 40.59 ms |

此表是分层同请求样本，不是全量监控均值；平台子阶段可能重叠，不能机械相加或全部解释为单程跨国网络耗时。部署均值接近包含内外层误差抵消，不能据此宣称时间模型准确。

## 采集说明与回退

真实事件只有mono_ns，必须使用process.json的wall_anchor_ns/mono_anchor_ns换算窗口。首次采集筛选错误产生的real0配对不是请求丢失，已修正后获得完整16/256配对。组间精简采集保留身份、接纳、批次成员、dispatch/ready/consume与输出事件；已用16条完整trace验证全部配对/token/时延字段完全一致。详细query轮询数据仍在源trace，必要时针对异常请求补取。

回退：真实应用恢复`-5aa8=1,-4b20=0`，仿真应用恢复`-chat=100,-cpu1=0`，打流目标恢复原`pp042/target.json`。不需要重载TPU模型。正式压测只对本次已提交的有限进程操作，不影响同事test3或线上部署。
