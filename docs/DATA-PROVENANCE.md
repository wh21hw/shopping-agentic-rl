# 数据溯源(DATA PROVENANCE)

本仓库所有训练/评测资产的开发机获取方式统一为:**有外网的机器下载 → 提交进 GitHub →
开发机从内网 git 拉取**。本文档记录每份数据"从哪来、怎么得到的、如何校验"。

## 1. 商品目录(环境运行必需)

| 项 | 值 |
|---|---|
| 文件 | `environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz`(19MB) |
| 解压后 | ~118MB JSON 数组,Catalog-Fine 全量商品(官方淘宝 2025-06 快照派生) |
| 来源 | 参考项目 [shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon) 内嵌的 ShopSimulator v2.1,随其仓库分发;与官方 HuggingFace 数据集 `wpei/ShopSimulator` 中的 `fine_items_eval_train_all.jsonl` 为同一份 catalog 的不同封装 |
| 出处声明 | `environments/ShopSimulator/EMBEDDED_SOURCE.json`(记录了上游 commit) |

> 官方 HF 数据集仅包含 catalog 分片(共 ~224MB),不包含任务文件;我们内嵌的 gz 已覆盖
> 环境运行所需,无需重复搬运。

## 2. 任务集(GRPO 训练 + 冻结评测)

| 文件 | 内容 |
|---|---|
| `data/grpo/train.jsonl` / `train.parquet` | 1000 道 GRPO 训练任务 |
| `data/grpo/validation.jsonl` / `validation.parquet` | 50 道验证任务 |
| `data/evaluation/tasks.jsonl` | 200 道冻结评测任务(`training_overlap: 0`) |

**关键点:这些文件只存 `task_id`,不存任务文本。** ShopSimulator 的任务由环境启动时
`get_goals()` 从商品目录确定性派生(每个商品的 `instruction` 字段),`task_id` 即 goals
列表索引。因此只要有第 1 节的 catalog,任务内容可完整还原,不存在"任务文件缺失"问题。

任务 ID 由参考项目从官方任务池中抽样冻结,每个目录的 `metadata.json` 记录 SHA-256
校验和与环境/奖励版本(`shopsimulator-environment-v2.1` / `shopsimulator-reward-v3`)。

## 3. SFT 教师轨迹(379 训练 + 49 验证)

| 项 | 值 |
|---|---|
| 文件 | `data/sft/train.jsonl`(10.2MB)/ `validation.jsonl`(1.4MB) |
| 来源 | **参考项目自采**,非官方发布(论文的 6K 条 GPT-4.1 轨迹未开源) |
| 生成方式 | 参考项目 `scripts/collect_sft_data.py`:Teacher 模型(默认 deepseek-v4-flash,OpenAI 兼容 API)在环境中 rollout → 验收过滤(`acceptance_reasons`)→ 排除 200 道留出评测题防泄漏 → 按 seed=42 九一切分 |
| 校验 | `data/sft/metadata.json` 含 train/validation 的 SHA-256 |

## 4. 模型权重(不进 git)

Qwen3.5-2B 基座权重放在 `models/` 目录(见 [models/README.md](../models/README.md)),
获取渠道与布局约定在该文件中说明。训练产出(SFT adapter、GRPO checkpoint)同样不进 git,
直接落在开发机本地。

## 5. 校验方法

开发机拉取仓库后可一键校验数据完整性:

```bash
# 校验和与 metadata.json 比对
sha256sum data/sft/train.jsonl data/sft/validation.jsonl \
          data/grpo/train.parquet data/grpo/validation.parquet \
          data/evaluation/tasks.jsonl
```

期望值见各目录 `metadata.json`。
