# Redis / V6D 拓扑核对（2026-11-17 20:04 北京时间）

结论：旧 Qwen3.7 多实例仿真使用 Redis tracker 做缓存位置发现；当前 test4 的真实与仿真均关闭 tracker 和 KVCM，保留原生 V6D 本地控制路径。当前不要求额外部署 Redis tracker，但这不是跨实例缓存能力已验证的证据。

| 项目 | high-pressure-qwen3.7-max-sim | test4 真实 | test4 仿真 |
|---|---|---|---|
| 推理部署 | -7350，20 实例 | 6842e3d9，PP4 | f3ded9ec，CPU1 |
| Redis 部署 | -0026，ubuntu:hisim-redis，1 实例 | 旧辅助 Redis 仍存在 | 无单独 Redis |
| DS_LLM_LAUNCH_V6D | 1 | 1 | 1 |
| DS_LLM_SHARE_V6D | 1 | 0 | 0 |
| V6D_ENABLE_TRACKER | 1 | 0 | 0 |
| V6D_KVCM_MODE | 本次未见显式值 | disabled | disabled |
| KVS_METASERVICE_REDIS_URI | 配置 | 保留外部 URI | 本机占位 URI |

`DS_LLM_SHARE_V6D` 控制同机多个引擎进程共享 V6D daemon，不能用它判断跨节点缓存索引是否开启。真实 PP4 的 connector 仍访问本机与另外三个 rank 的 V6D，这属于同一模型副本内部的分片协调，并非 Redis tracker 查找另一个 Prefill 副本。

当前真实容器的源码证据：

- `/opt/dashllm-venv/lib/python3.12/site-packages/dashllm/utils/vineyard.py:288`：只有 `V6D_ENABLE_TRACKER` 为真才传入 `--tracker-redis` 与部署命名空间；299 行只有 KVCM 模式非 disabled 才传 manager endpoint。
- 运行中的 V6D PID399 命令行没有 `--tracker-redis` 或 KVCM 参数。
- `v6d/cli/serve.py:332`：没有 tracker 参数时配置默认 `redis://localhost:6379`。
- `v6d/server/peers/tiered_vineyard/peer.py:961`：初始化时创建 tracker 并 ping，失败返回 None；不应表述为完全没有启动连接尝试。
- 当前真实实例在 **2026-11-17 03:03:57** 的日志：`Tracker ping failed for type=memory, running without it`；随后 `Using TieredVineyardPeer (sharedfs=disabled, tracker=disabled, eviction=memory_usage)`。
- 上层 `_backend_vllm.py` 将 `DS_LLM_LAUNCH_V6D=1` 且 URI 非空作为 `_use_v6d` 条件；因此保留 URI 不等于运行中逐请求查询该 Redis，直接删 URI 反而可能关闭 V6D。

共享 Chat 另有增强缓存/理论命中率相关 Redis 配置；它们与 V6D 的跨实例位置索引不同。本次未逐命令采集 Chat Redis 流量，不能声称整条请求链完全不访问 Redis。

精度覆盖限制：上一轮 p3r2-main 三档的两侧共 30179 次 V6D 查询均 miss，约 35% 前缀命中主要来自 HBM。现有结果验证 HBM 与 V6D miss/元数据路径，未验收 DRAM 命中加载和跨实例缓存命中。后续应补超过 HBM 保留范围、仍在 V6D 内的定向复用用例；若恢复跨实例缓存，再成对启用 tracker/KVCM 并验证独立缓存命名空间。

旧应用：https://dashscope-spectrum.alibaba-inc.com/console/workspaces/464482ce/applications/high-pressure-qwen3.7-max-sim
