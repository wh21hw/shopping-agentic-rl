# Shop Agent RL

基于 [ShopSimulator](https://github.com/ShopAgent-Team/ShopSimulator) 中文购物环境的
Shopping Agent 后训练系统:**LoRA SFT 冷启动 → veRL 在线 GRPO → 冻结 Benchmark 评测 →
Badcase 驱动的算法改进(DAPO)**。

> 开发计划与进度见 [PLAN.md](PLAN.md);数据与权重的来源见 [docs/DATA-PROVENANCE.md](docs/DATA-PROVENANCE.md);
> 内嵌资产出处见 [docs/reference/ATTRIBUTION.md](docs/reference/ATTRIBUTION.md)。

## 目标

```text
Qwen3.5-2B (Base, 严格成功率 ≈ 0%)
    → LoRA SFT(教师轨迹冷启动)
    → veRL GRPO(在线 rollout + 确定性 Reward v3)
    → 冻结 200 题评测:Base / SFT / GRPO / DAPO 对比
```

## 仓库结构

```text
shop-agent-rl/
├── PLAN.md                       # 七天开发计划与共识
├── environments/
│   └── ShopSimulator/            # 内嵌 ShopSimulator Environment v2.1 源码 + 商品数据(19MB gz)
├── data/
│   ├── sft/                      # 379 条训练 + 49 条验证教师轨迹(action-only)
│   ├── grpo/                     # GRPO 训练/验证任务(jsonl + parquet)
│   └── evaluation/               # 冻结的 200 道留出评测任务
├── models/                       # 模型权重存放处(仅 README 入库,权重不进 git)
├── docs/
│   ├── DATA-PROVENANCE.md        # 数据集/权重溯源与完整性校验
│   ├── research/                 # Day 0 上游研究(论文精读 + 两仓库调研 + 继承关系)
│   └── reference/                # 参考设计文档(Reward v3 / SFT / GRPO / 评测,注明出处)
└── rl_train/                     # 本项目自研代码(环境封装 / Reward / AgentLoop / 评测)
    ├── configs/                  # 训练与评测配置
    ├── data/                     # 数据审计与转换脚本
    ├── env/                      # ShopEnvironment 客户端 / RL wrapper / action parser
    ├── agent_loop/               # veRL AgentLoop 适配层
    ├── rewards/                  # Reward v3 独立封装 + veRL adapter
    ├── eval/                     # 统一评测 / metrics / badcase 分析
    ├── tests/                    # 最小测试
    └── scripts/                  # 一键入口(start_env / sft / grpo / eval)
```

## 硬件与环境

| 项 | 值 |
|---|---|
| GPU | 2 × A100 80GB(内网开发机) |
| 镜像 | veRL 官方镜像:CUDA 12.4.131 / Python 3.10.13 / torch 2.6.0 / vLLM 0.8.5.post1 / veRL 0.3.1.dev0 |
| 基座模型 | Qwen3.5-2B |

## 致谢与出处

环境与数据来自 [ShopSimulator](https://github.com/ShopAgent-Team/ShopSimulator)
([arXiv:2601.18225](https://arxiv.org/abs/2601.18225));
管线设计与参考实现来自 [shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon);
训练框架为 [veRL](https://github.com/verl-project/verl)。详见 ATTRIBUTION.md。
