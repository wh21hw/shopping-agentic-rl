# models/ — 模型权重存放约定

权重文件 **不进 git**(见根目录 `.gitignore`),本目录仅在开发机上实际存在。
本机/开发机通过各自的渠道获取权重后,按以下布局放置:

```text
models/
├── Qwen3.5-2B/                  # 基座模型(HF 原始权重,safetensors)
│   ├── config.json
│   ├── tokenizer.json
│   └── *.safetensors
├── sft-lora/                    # Day 3 LoRA SFT 产出的 adapter(也可放合并后的全量权重)
│   └── ...
└── grpo/                        # Day 4-5 GRPO 训练 checkpoint / 合并权重
    └── ...
```

## 获取渠道

| 权重 | 渠道 |
|---|---|
| Qwen3.5-2B | 优先开发机内网模型仓库/镜像源;否则在有外网的机器从 HuggingFace / ModelScope 下载后按内网传输流程带入 |
| SFT / GRPO 产出 | 训练直接产出到开发机本目录或 `checkpoints/`,不需要跨机器传输 |

## 路径引用约定

训练与评测脚本一律使用相对仓库根目录的路径 `models/Qwen3.5-2B`,
或在 `rl_train/configs/` 中集中配置,避免硬编码绝对路径。
