# 内嵌资产出处声明(ATTRIBUTION)

本仓库内嵌的第三方资产均**非本仓库原创**,特此声明出处与许可状态。
使用前请遵守各上游项目的原始许可证。

| 路径 | 来源 | 说明 |
|---|---|---|
| `environments/ShopSimulator/` | [YYHDBL/shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon) 内嵌的 ShopSimulator Environment v2.1 | 环境源码最初来自 [ShopAgent-Team/ShopSimulator](https://github.com/ShopAgent-Team/ShopSimulator)([arXiv:2601.18225](https://arxiv.org/abs/2601.18225));含商品目录 `shop_env/data/fine_items_eval_train_all.json.gz` |
| `data/sft/` | 同上 | 379 条训练 + 49 条验证的教师模型轨迹(action-only,已删除教师私有推理),经 Reward v3 终局验收 |
| `data/grpo/` | 同上 | GRPO 训练(1,000 任务)与验证(50 任务)任务集,含 metadata 哈希 |
| `data/evaluation/` | 同上 | 冻结的 200 道留出评测任务 |
| `docs/reference/reward-v3.md` | 同上 docs/ | Reward v3 设计文档 |
| `docs/reference/sft.md` | 同上 docs/ | LoRA SFT 设计文档 |
| `docs/reference/grpo.md` | 同上 docs/ | veRL GRPO 集成文档(注意:按 verl 0.8.0 编写,本项目镜像为 0.3.1.dev0,接口需现场适配) |
| `docs/reference/evaluation.md` | 同上 docs/ | 评测流水线设计文档 |
| `docs/reference/data-collection.md` | 同上 docs/ | 教师轨迹采集文档 |

## 与上游的关键差异

- **veRL 版本**:参考实现钉死 `verl==0.8.0` 并附带 SHA-256 校验补丁;本项目运行环境为
  veRL 官方镜像内置的 `0.3.1.dev0`,AgentLoop 适配层需按实际安装源码重新适配,
  **参考项目的补丁不能直接套用**。
- **硬件**:参考实现按单张 RTX 6000 96GB 验证;本项目为 2 × A100 80GB。

## 引用

```bibtex
@misc{wang2026shopsimulatorevaluatingexploringrldriven,
      title={ShopSimulator: Evaluating and Exploring RL-Driven LLM Agent for Shopping Assistants},
      author={Pei Wang and others},
      year={2026},
      eprint={2601.18225},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.18225}
}
```
