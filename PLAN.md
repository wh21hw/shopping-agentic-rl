# Shop Agent RL — 七天开发计划

> 一周内完成,作为简历项目。目标:基于 ShopSimulator 构建可复现的 Shopping Agent 后训练系统
> (LoRA SFT → veRL 在线 GRPO → 冻结 Benchmark 评测 → Badcase 驱动的算法改进)。

## 已锁定共识

| 项 | 决定 |
|---|---|
| 基座模型 | Qwen3.5-2B(与参考项目同基座,配置可复用) |
| 训练管线 | SFT → GRPO 两段式(实测 Base 成功率 0%,SFT 冷启动不可跳过) |
| 硬件 | 内网开发机:2 × A100 80GB,veRL 官方镜像 |
| 镜像环境 | CUDA 12.4.131 / Python 3.10.13 / torch 2.6.0 / vLLM 0.8.5.post1 / **veRL 0.3.1.dev0** |
| 代码协作 | 本机 → GitHub → 内网 git → 开发机;开发机无外网 |
| Checkpoint | 存开发机项目文件夹,不进 git;仅曲线/metrics/最终模型信息回传 |
| 交付物 | GitHub 仓库 + 对比表 + 训练曲线 + README + Badcase 分析 + 带故事的算法改进 |
| 依赖获取 | 开发机内网 pip 源可用(已确认) |

## 参考项目事实(调研结论)

- **ShopSimulator 官方仓库**(ShopAgent-Team/ShopSimulator,commit `51bb260`):纯环境 + 评测,
  **无任何 RL 训练代码**(全仓库 grep 确认)。
- **shopping-grpo-longhorizon**(YYHDBL):工程完善的 agentic RL 实现——钉死 verl==0.8.0、
  SHA-256 校验补丁、自研 AgentLoop 适配层、有界动态采样、27 个测试、冻结 200 题 benchmark。
  实测结果:Qwen3.5-2B Baseline 严格成功率 **0.0%** → LoRA SFT **60.5%** → GRPO step 100 **62.0%**。
  单张 RTX 6000 96GB:SFT 3 epoch ≈ 3h;GRPO 100 步 ≈ 3–4h(~110s/步)。
- 本项目已内嵌其 ShopSimulator v2.1 环境源码、全部数据与关键设计文档(见 `docs/reference/`,出处见 `docs/reference/ATTRIBUTION.md`)。

## ⚠️ 关键技术风险

1. **veRL 版本落差**:参考项目按 `verl==0.8.0` 开发,开发机镜像是 **0.3.1.dev0**。
   AgentLoop / rollout API 两版差异大,SHA-256 补丁无法直接套用。
   **规则:一切以开发机实际安装的 verl 源码为准,不照搬 0.8.0 的适配代码与补丁。**
2. **模型权重获取**:开发机无外网,Day 1 第一项必须确认 Qwen3.5-2B 权重的内网获取渠道
   (内网 ModelScope/HF 镜像);若无,需本机下载经内网 git 传输(~5GB,注意仓库大小限制)。
3. **GRPO 夜间训练失败**:Day 4 内置断点续训验证;最坏情况改跑 100 步短版,牺牲步数保交付。

## 范围边界(场景覆盖)

论文的四个场景:单轮/多轮 × 标准/个性化。本项目只覆盖**单轮 × 标准**:

| 场景 | 环境代码支持 | 参考项目 | 本项目 | 说明 |
|---|---|---|---|---|
| 单轮 × 标准 | ✅ | ✅ | ✅ | 全部 1250 道题(1000+50 训练/验证 + 200 评测)均属此场景 |
| 单轮 × 个性化 | ✅(API 有 `if_persona`) | ❌ | ❌ | 环境支持但无对应任务与轨迹,不在一周范围内 |
| 多轮 × 标准(用户反馈/追问) | ❌ 需用户模拟器(官方 `multi_eval/`,未内嵌) | ❌ | ❌ | Future Work |
| 多轮 × 个性化 | ❌ 同上 | ❌ | ❌ | Future Work |

多轮场景需要 LLM 扮演模拟用户给出追问/反馈,配套用户模拟器 + 多轮 reward 归因,
工程量是另一个量级。列入 Future Work(与 OPD 并列),不影响一周交付的故事完整性。

---

## Day 0 — 上游研究(已完成)

- [x] 精读论文 arXiv:2601.18225(环境构建 / reward 公式 / 评测 / RL 探索结论)
- [x] 调研官方仓库(API 协议 / 动作格式 / get_score 指标 / 无训练代码的事实)
- [x] 调研参考项目(Action Guard / 工具化动作 / AgentLoop / Reward v3 / 动态采样)
- [x] 明确本仓库继承关系与自研范围
- 📄 产出:[docs/research/day0-upstream-research.md](docs/research/day0-upstream-research.md)
- 🔑 指导后续的关键结论:① SFT+RL 优于单独 RL;② strict reward 作 RL 目标优于 loose;
      ③ 单轮优先;④ badcase 重点盯 attribute/option 两维

## Day 1 — 仓库地基 + 环境闭环(无 GPU)

- [x] 建仓,导入内嵌 ShopSimulator v2.1 环境 + 数据 + SFT 轨迹 + 冻结 200 题(注明来源)
- [x] 建 `rl_train/` 骨架、`.gitignore`、README、PLAN.md
- [ ] **开发机前置核查**:① Qwen3.5-2B 权重内网获取渠道;② 内网 pip 源(已确认可用);③ 记录镜像 veRL 0.3.1.dev0 的 AgentLoop 实际接口
- [ ] 环境冒烟:开发机上启动 env 服务,脚本跑通 reset → search → click → buy
- ✅ 验收:仓库推到 GitHub;开发机拉取成功;env 服务一条命令启动;核查项有明确结论

## Day 2 — Env Wrapper + Reward + 数据审计(纯 CPU)

- [ ] 数据审计:train/eval schema、**train ∩ eval overlap 必须为 0**、产出 DATASET.md
- [ ] Action parser + ≥15 个测试用例(复用参考项目 action 格式)
- [ ] 封装 Reward v3,与参考实现/官方评分逻辑对拍,**mismatch = 0**
- [ ] `ShopEnvironment` RL 风格包装(reset/step/done/info、max_steps、invalid action 不崩)
- ✅ 验收:全部测试绿;无 LLM 完成 task → actions → reward 闭环

## Day 3 — SFT 冷启动

- [ ] 用内嵌 379 条教师轨迹做 LoRA SFT(action-only mask)
- [ ] 双卡 A100 训练(预计 1–2h),merge 权重
- [ ] 20 题 sanity 评测
- ✅ 验收:sft-merged 可用,小样本成功率明显 > 0(参考项目全量 60.5%)

## Day 4 — veRL GRPO 打通(核心日)

- [ ] 按 **veRL 0.3.1.dev0 实际接口**实现/移植 AgentLoop 适配层 + 工具层 + 动态采样
- [ ] 构建 parquet 训练数据 → `--dry-run` → 20 步 smoke test
- [ ] **断点续训验证**(硬需求)
- [ ] 启动正式训练(100–500 步,挂夜)
- ✅ 验收:global_step ≥ 20,reward 有记录,resume 验证通过,夜间训练在跑

## Day 5 — 评测固化 + 简历核心产出

- [ ] 检查夜间训练,按 validation 指标选 checkpoint,export actor
- [ ] 统一评测管线:Base / SFT / GRPO × 冻结 200 题(每组约 40–60min)
- ✅ 验收:三模型对比表 + step→reward/success/invalid 曲线(**简历核心产出**)

## Day 6 — Badcase 分析 + 带故事的改进

- [ ] Badcase analyzer(invalid action / 重复搜索 / 约束违反 / 从未 buy / 步数耗尽等),
      对比 Base vs GRPO 错误分布变化
- [ ] **故事线**:从 badcase 定位 GRPO 短板(低熵坍缩 / clip 截断 / 零 advantage batch)
      → 引出 DAPO(clip-higher + 动态采样 + token-level loss)
- [ ] 启动 DAPO 训练挂夜(100–200 步)
- ✅ 验收:badcase 报告 + 改进动机文档 + 训练在跑

## Day 7 — 收尾 + 简历交付(兼缓冲日)

- [ ] DAPO 评测 → 四模型完整对比表(Base / SFT / GRPO / DAPO)
- [ ] README 完善:架构图、结果表、一键复现命令、局限与 Future Work(OPD、多轮/个性化场景等)
- [ ] 面试讲稿:每个决策一句话 + 关键数字
- 🛟 本天同时承担 Day 4–6 延误的缓冲
- ✅ 验收:仓库公开可复现,故事线完整

## 开发铁律

1. 不修改官方/参考环境源码语义;新代码进 `rl_train/`
2. 严禁用评测集训练;train/eval overlap = 0
3. 一切 veRL 接口以开发机安装版本源码为准,不猜
4. 每个功能带最小测试;所有实验固定 seed;评测用固定 task IDs
5. 训练必须可 resume;checkpoint 不进 git
6. 一个 Task 一个 commit,不提交权重/日志/缓存
