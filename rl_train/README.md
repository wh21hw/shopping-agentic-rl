# rl_train/ — 本项目自研代码

所有本项目新增代码放在此目录,**不修改上游环境源码语义**。

## 目录职责

| 目录 | 职责 | 依赖约束 |
|---|---|---|
| `configs/` | SFT / GRPO / DAPO / eval 配置 | — |
| `data/` | 数据审计(inspect)、schema 文档、veRL parquet 构建 | 禁依赖 veRL |
| `env/` | ShopSimulator HTTP 客户端、RL 风格 wrapper、action parser、trajectory schema | 禁依赖 veRL / transformers / vLLM |
| `agent_loop/` | veRL AgentLoop 适配层(rollout ↔ 环境多步交互) | 以镜像实际 verl 版本接口为准 |
| `rewards/` | Reward v3 独立封装 + veRL reward adapter | 封装本体禁依赖 veRL |
| `eval/` | 统一评测、metrics 聚合、badcase 分析 | — |
| `tests/` | 各模块最小测试 | — |
| `scripts/` | 一键入口:start_env / sft / grpo / eval | — |

## 关键规则

1. train ∩ eval overlap 必须为 0(构建数据集时 assert)
2. 评测固定使用 `data/evaluation/tasks.jsonl` 的 200 个 task_id
3. Reward 语义以上游 Reward v3 为准,禁止隐式修改
4. invalid action 不允许 crash 环境
5. 所有实验固定 seed;所有 trajectory 可保存为 JSONL
6. 训练必须可 resume(checkpoint 存开发机本地,不进 git)
