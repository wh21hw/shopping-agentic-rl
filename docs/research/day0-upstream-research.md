# Day 0 上游研究报告:ShopSimulator 论文 + 两个仓库全面调研

> 研究日期:2026-08-09
> 对象:① 论文 arXiv:2601.18225(阿里/淘宝团队);② 官方仓库 ShopAgent-Team/ShopSimulator(commit `51bb260`);③ 参考实现 YYHDBL/shopping-grpo-longhorizon
> 目的:在动手前彻底吃透环境、reward、评测与训练管线,明确本仓库的继承关系。

---

## 第一部分:淘宝论文精读(arXiv:2601.18225)

### 1.1 这篇论文在做什么

阿里团队构建了一个**中文购物 Agent 沙盒**:既能评测,也能训练(这是它与 WebShop 等纯 benchmark 的最大区别)。核心问题:LLM 能不能当一个靠谱的购物助手——理解模糊需求、多轮澄清、利用用户画像、在上百个高度相似商品里精确选中正确的那个?

### 1.2 环境怎么建的(三段流水线)

**① 商品目录(真实淘宝数据)**
- 取 2025 年 6 月淘宝快照,按曝光频率选 Top 5000 万商品为初始池
- 过滤:砍掉子类少于 10 个的一级类目;用 GPT-4o 评估类目复杂度/多样性,低复杂度剔除、低多样性减采样
- **Catalog-Fine**:每个最小子类保留约 **120 个高度相似商品**,共约 **2 万**件——这是"细粒度区分"难度的来源(不能靠随机猜)
- **Catalog-Full**:另外 130 万件,覆盖广度;所有实验用 Catalog-Fine(延迟考虑)
- 12 个大域(domain)

**② 任务构造(28K 任务)**
- 每个自然语言购买指令与**唯一 gold 商品一对一绑定**,保证评测无歧义
- 人工标注员写指令,**禁止抄商品标题/关键词**,必须用同义替换、描述性短语、口语化表达(如"轻一点的"而不是具体克数)——逼模型做语义理解而非字符串匹配
- 标注员还要验证没有其他相似商品能完全满足该指令
- 25K 训练任务 + 2.8K 评测任务

**③ 用户建模(个性化 + 多轮)**
- **个性化**:4,726 个结构化用户画像(年龄、性别、消费层级、品牌偏好)。关键设计:把画像相关信息**从指令里删掉**,只留商品特定线索——防止模型走"只读指令"的捷径
- **多轮**:LLM 扮演 shopper,**故意从模糊需求开始**,只在 agent 追问时才透露属性——考察主动澄清能力

### 1.3 Reward 设计(论文公式,官方代码已验证一致)

设用户要求类目 \(U_{cat}\)、属性集 \(U_{att}\)、规格集 \(U_{opt}\)、价格上限 \(U_{price}\),购买商品 \(y\):

**R_loose(加法式,沿用 WebShop)**:
\[
R_{loose} = R_{cat} \cdot \frac{|U_{att} \cap Y_{att}| + |U_{opt} \cap Y_{opt}| + \mathbb{1}[Y_{price} \le U_{price}]}{|U_{att}| + |U_{opt}| + 1}
\]

**R_strict(乘法式,瓶颈原则)**:
\[
R_{strict} = r_{cat} \cdot \frac{|U_{att} \cap Y_{att}|}{|U_{att}|} \cdot \frac{|U_{opt} \cap Y_{opt}|}{|U_{opt}|} \cdot \mathbb{1}[Y_{price} \le U_{price}]
\]

**R_succ**:四个维度全部完美匹配才算 1。

代码层面(官方 `get_score.py` / `goal.py` 实测):
- `r_type ∈ {1.0, 0.5}`:query 相同 / 类目交集 ≥2 层 / 标题名词重合 >0.2,满足任一得 1.0,否则 0.5
- `r_att`:每个目标属性用 `fuzz.token_set_ratio > 85` 模糊匹配,或出现在标题/卖点/描述中
- `r_option`:同样模糊匹配(颜色等先归一化)
- `r_price`:布尔值,`price ≤ price_upper`(上限从高于商品价的离散价格档随机抽两个取大)
- 官方指标:`r_hard = r_type × r_att × r_option × r_price`;`r_success = 四项全 1`

### 1.4 评测结果(有多难)

| 模型 | Single-Turn R_succ | Overall R_succ |
|---|---:|---:|
| GPT-5 | 40.78% | **32.65%(最佳)** |
| DeepSeek-V3.1 | 31.86% | 31.81% |
| Qwen3-235B-A22B | 27.96% | 25.66% |
| Qwen3-8B | 14.13% | 12.87% |

- 所有模型 R_loose ≫ R_strict ≫ R_succ:能买对大类,但**细粒度属性/规格普遍满足不了**
- 多轮场景全面掉点;小模型掉得更狠(Qwen3-8B 单轮→多轮 strict 掉 47%)

**错误分析(Claude-4-Sonnet 轨迹,GPT-5 辅助分类)**:
- **Search**:忽略关键属性、放弃高匹配结果、重复查询(状态记忆差)
- **Click**:违反用户显式约束(约束执行弱)
- **BuyNow**:近 **80%** 错误是"没确认细节就买/用户明确拒绝还买"(决策鲁莽)
- **个性化**:信息利用不足 55.8% vs 过度解读 35.7%——两头失衡

### 1.5 RL 训练探索(对我们最有价值的部分)

**设置**:Qwen3-8B;SFT 用 **GPT-4.1 采集的 6K 条成功轨迹**(bs 32,lr 1e-5,4 epoch);RL 用 **ROLL 框架 + GRPO,去掉 KL loss 鼓励探索**,32K 上下文,lr 1e-6,200 步,每步 32 样本 × 8 rollout。

**结果(Single-Turn R_succ)**:Baseline 14.13 → SFT 32.47 → RL(strict) 30.19 → **SFT+RL(strict) 38.89**。四个场景全部是 SFT+RL(strict) 最优。

**五个核心结论(直接指导我们的项目)**:

1. **SFT 与 RL 互补,SFT+RL 全面优于单独 RL**。无先验时 RL 自主探索极难(尤其多轮);SFT 注入工作流先验后,RL 再学偏好、补短板。→ 支持我们"两段式"路线。
2. **strict reward 作 RL 目标全面优于 loose**。瓶颈式乘法 reward 把优化压力集中到最弱维度(属性/规格);loose 训出来的模型偏向"覆盖率/完成任务",准确性差。→ 我们 GRPO 应使用 strict 风格 reward(参考项目 Reward v3 正是此类)。
3. **SFT 学流程,RL 学偏好**:SFT 模仿教师的长轨迹(轨迹也变长),个性化错误不降;RL 自探索轨迹短,且在单轮个性化场景**反超 SFT +8.25%**。
4. 单轮场景训练收益大;**多轮训完仍只有 ~35%**,长程一致性是未解难题。→ 我们第一阶段只做单轮是正确取舍。
5. 增益主要来自 attribute 和 option 匹配的提升。→ badcase 分析应重点盯这两维。

---

## 第二部分:官方仓库调研(ShopAgent-Team/ShopSimulator)

### 2.1 结构与职责

```
shop_env/          环境服务(Flask HTTP)+ 数据
  shop_env/pack_api.py    服务入口,监听 127.0.0.1:5000
  web_agent_site/engine/  核心引擎:goal.py(reward)、engine.py(状态机)、normalize.py
single_eval/       单轮评测(agent.py 走 OpenAI 兼容 API 调模型)
multi_eval/        多轮评测(多了 shopper.py 模拟用户)
get_score.py       指标聚合
```

### 2.2 环境 API 协议(我们 Day 1/2 要对接的)

单一端点 `POST /api/shop_agent`,三种 action:
| action | 请求体 | 返回 |
|---|---|---|
| `reset` | `{"action":"reset","idx":task_id}` | `env_idx` + `instruction`(用户需求)+ 环境初始状态 |
| `interact` | `{"action":"interact","env_idx":...,"response":action_text}` | 新 observation;结束时带 `done/over` + `reward` + `reward_detail` + `goal` + `purchase` |
| `release_one` | `{"action":"release_one","env_idx":...}` | 释放环境实例 |

### 2.3 动作协议与 System Prompt

- 动作格式(WebShop 风格文本):`search[关键词]`、`click[值]`;购买 = `click[buy now]`,且**购买前必须先点过"商品规格"按钮选好规格**
- 官方 system prompt 规定 ReAct 式输出:`Thought: ... → Action: click[某值]`,一轮一个动作
- single-turn standard 评测集 task_nums = **1495**;多轮 1459
- **依赖**:spacy(`zh_core_web_sm`)+ thefuzz(reward 模糊匹配必需)

### 2.4 关键事实

- **仓库没有任何 RL 训练代码**(全仓库检索确认)。论文里的 ROLL+GRPO 训练没有开源;`agent.py` 只支持阿里内部 idealab API——内部依赖是训练代码不开源的原因之一。
- **任务数据不在仓库里**,在 HuggingFace(`wpei/ShopSimulator`)。由于我们只能从 GitHub 获取数据,官方数据链路不可用 → 采用参考项目内嵌的任务集(见继承关系)。

---

## 第三部分:参考项目调研(shopping-grpo-longhorizon)

### 3.1 定位

对官方环境的**完整 agentic RL 后训练实现**:教师轨迹采集 → LoRA SFT → veRL 0.8 GRPO → 冻结 200 题四面板评测。内嵌 ShopSimulator **v2.1**(比官方仓库版本新,含 reward.py / termination.py / comparators.py 等增强模块)。

### 3.2 值得抄的设计(按重要性)

**① Action Guard(`environment/actions.py`)**——动作在到达环境前先校验:
- 点击目标必须存在于**最新 observation** 的可点按钮/商品 ASIN 中(按钮只在当前页有效)
- 拦截 schema 外参数、非法 finish 理由、搜索不可用页面上的 search
- 拒绝时返回结构化 tool error(附当前页合法目标列表),让模型自我纠正
- 数据佐证:Baseline guard 拒绝 752 次 → SFT 52 → GRPO 38

**② 工具化动作(OpenAI tool-calling schema)**:`think`、`search_products`、`open_product(asin)`、`select_option`、`buy_now`、`finish_without_purchase(reason=no_suitable_product)`、`prev_page`、`back_to_search`。比官方裸文本 `click[...]` 更结构化、更易训练。

**③ AgentLoop(veRL `ToolAgentLoop` 子类)**:
- 上下文预算管理:24576 窗口,输入预算 16384,可选历史压缩(compaction)
- Observation 投影:搜索结果/商品详情各有 token 预算(1536/4096/768),防长 observation 撑爆上下文
- 终局 reward 两种模式:`native`(官方风格)/ `constraint_aware`(Reward v3);可选长度 shaping(默认关)
- `max_steps=35`,env 端口 5700

**④ 有界动态采样(`dynamic_sampling.py` + SHA-256 校验补丁)**:全组同 reward(advantage=0)时重采样,上限 3 个 batch、最多连续跳过 10 次更新——防止无限重采样循环。

**⑤ Reward v3(确定性终局 reward)**:类目 + 预算是硬门槛;品牌/型号/功能/规格按 0.35/0.25/0.25/0.15 加权;命中 gold 商品 1.0,合格替代品 0.55,部分满足 ≤0.25;错误购买/过早放弃/循环/超步数给负分;证据不足标 `reward_valid=false` 而不是伪装成 0 分。

**⑥ SFT 数据策略**:教师轨迹只保留 action(删除教师私有推理),loss 只在 assistant action token 上计算,mask 掉 observation——学"可执行策略"而非背诵环境输出。

### 3.3 它的实测数字(我们的对照基准)

| 模型 | Done | Strict 成功 | 平均 Reward | 平均步数 | Guard 拒绝 |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-2B Base | 18.0% | 0.0% | -0.1105 | 5.9 | 752 |
| LoRA SFT | 96.5% | 60.5% | 0.4729 | 12.3 | 52 |
| GRPO step 100 | 96.5% | **62.0%** | 0.5158 | 11.9 | 38 |

训练配置:LoRA rank 16/alpha 32,lr 1e-6,rollout 4/prompt,temp 0.7,top-p 0.9,KL 全关,batch 2,500 步上限;单卡 RTX 6000 96GB,GRPO ~110s/步。

### 3.4 工程亮点(我们要学的习惯)

依赖钉版本 + 补丁带 SHA-256 校验、数据带 metadata 哈希、评测四面板不合成分、27 个测试文件、`--dry-run` 先行、experiments/ 存机器可读配置与结果。

---

## 第四部分:本仓库的继承关系

### 4.1 继承自淘宝官方(ShopSimulator)

| 内容 | 形式 |
|---|---|
| 环境交互范式(reset/interact、search/click 动作、observation 驱动) | 设计继承;代码实际采用参考项目内嵌的 v2.1 版 |
| Reward 维度定义(类目/属性/规格/价格)与指标体系(r_loose/r_hard/r_success) | 概念继承,评测报告沿用 |
| 论文结论(SFT+RL、strict reward、单轮优先) | 直接指导本项目技术路线 |

### 4.2 继承自参考项目(shopping-grpo-longhorizon)

| 资产 | 位置 | 状态 |
|---|---|---|
| ShopSimulator v2.1 环境源码 + 商品目录 | `environments/ShopSimulator/` | 已入仓,不改语义 |
| SFT 教师轨迹(379+49 条) | `data/sft/` | 已入仓 |
| GRPO 任务集(1000 训练 + 50 验证) | `data/grpo/` | 已入仓 |
| 冻结 200 题评测集 | `data/evaluation/` | 已入仓,评测分母固定 |
| Reward v3 / SFT / GRPO / 评测 / 数据采集设计文档 | `docs/reference/` | 已入仓 |
| Action Guard、工具 schema、observation 投影、动态采样、SFT mask 策略 | 设计参考 | Day 2–4 按 **veRL 0.3.1.dev0** 重新实现 |

### 4.3 本仓库自研部分(全部进 `rl_train/`)

1. **veRL 0.3.1.dev0 AgentLoop 适配层**(参考项目按 0.8.0 写,不能照搬——最大工程风险)
2. 环境 RL wrapper + action parser + 一致性测试
3. Reward v3 独立封装 + 对拍验证(mismatch=0)
4. 统一评测管线(Base/SFT/GRPO/DAPO × 200 题)+ 训练曲线
5. Badcase 分析器(对齐论文错误分类:search/click/buy 三类错误 + 个性化失衡)
6. **DAPO 改进实验**:从 badcase 证据出发选型(clip-higher + 动态采样 + token-level loss)

### 4.4 诚实性说明(简历表述口径)

- 环境与数据资产来自上游,`docs/reference/ATTRIBUTION.md` 已声明出处
- 本项目贡献:**在 veRL 0.3.x 上独立重建 agentic RL 训练管线** + 双卡 A100 适配 + badcase 驱动的算法改进实验,而非复现参考项目本身
