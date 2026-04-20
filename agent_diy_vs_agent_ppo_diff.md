# agent_diy 与 agent_ppo 详细对比说明

本文按文件逐个对照 [agent_diy](agent_diy) 与 [agent_ppo](agent_ppo) 的实现差异，重点说明两者在策略建模、特征工程、奖励设计、训练流程与配置上的不同。整体上看，`agent_ppo` 更接近一个结构简洁、输入维度较大、奖励逻辑较直接的 PPO 基线；`agent_diy` 则是围绕更紧凑的观测、更强的奖励塑形、更大的网络容量和更稳健的工程细节做过增强的版本。

## 总体结论

两套代码的共同点非常多：都采用相同的 PPO 基本框架、相同的训练工作流、相同的监控面板，也都使用 `BaseAgent`、`Algorithm`、`Model`、`Preprocessor` 这条主线。真正决定性能差异的部分主要集中在三处：

1. 观测空间不同。`agent_ppo` 使用 546 维输入，保留了更大范围的物件与局部地图信息；`agent_diy` 使用 107 维输入，把信息压缩到更聚焦的决策特征上。
2. 奖励设计不同。`agent_ppo` 主要依赖生存、距离 shaping、收集奖励和物件势能 shaping；`agent_diy` 则把奖励拆得更直接，强调生存、距离变化、收集、闪现逃脱和方向偏好。
3. 模型容量不同。`agent_ppo` 是两层 MLP，`agent_diy` 是三层 MLP，容量更大，和更紧凑的输入做了配套。

## 主要差异

### 1. [feature/preprocessor.py](agent_diy/feature/preprocessor.py#L51) vs [feature/preprocessor.py](agent_ppo/feature/preprocessor.py#L69)

#### 4.1 特征维度和组成不同

`agent_ppo` 的特征总长是 546 维，组成如下：

- 英雄自身 4D。
- 怪物 5D × 2。
- 物件 72D。
- 局部地图 441D。
- 合法动作 16D。
- 进度 3D。

对应配置在 [agent_ppo/conf/conf.py](agent_ppo/conf/conf.py#L17)。

`agent_diy` 的特征总长是 107 维，组成如下：

- 英雄自身 6D。
- 怪物 7D × 2。
- 宝箱 12D。
- buff 8D。
- 局部地图 49D。
- 合法动作 16D。
- 进度 2D。

对应配置在 [agent_diy/conf/conf.py](agent_diy/conf/conf.py#L31)。

#### 4.2 英雄特征不同

`agent_ppo` 只保留：

- 位置 x、z。
- 闪现冷却。
- buff 剩余时间。

`agent_diy` 额外加入：

- buff 是否激活。
- 当前速度归一化。

这意味着 `agent_diy` 对英雄当前状态的刻画更完整，尤其能区分“buff 存在但未激活”和“buff 正在生效”这类细节。

#### 4.3 怪物特征不同

`agent_ppo` 的单个怪物特征是 5D：

- 是否可见。
- 位置 x、z。
- 速度。
- 距离。

`agent_diy` 的单个怪物特征是 7D：

- 是否可见。
- 位置 x、z。
- 速度。
- 距离。
- 相对方向。
- 是否威胁。

也就是说，`agent_diy` 不只告诉模型“怪物在哪”，还告诉模型“这个怪物是不是迫在眉睫”，这会让策略更容易学到避险优先级。

#### 4.4 物件建模方式完全不同

`agent_ppo` 使用一个统一的 72D 物件槽位，编码的是通用 organ 信息：

- sub_type。
- status。
- x。
- z。
- 距离。
- 相对方向。

并且它还会对 organs 按 `config_id` 排序，以减少跨步抖动。

`agent_diy` 则把物件拆成了两类：

- 宝箱 12D，最多 3 个，每个 4 维。
- buff 8D，最多 2 个，每个 4 维。

这种做法的优点是语义更清晰。模型一眼就知道哪些槽位是宝箱，哪些是 buff，不需要自己从通用 organ 向量里再分类型。

#### 4.5 局部地图范围不同

`agent_ppo` 用 21 × 21 的局部地图窗口，共 441D。

`agent_diy` 只用 7 × 7 的中心补丁，共 49D。

这代表两种不同取舍：

- `agent_ppo` 更重视空间上下文和更大范围的障碍布局。
- `agent_diy` 更强调决策中心区域，减少输入噪声。

#### 4.6 进度特征不同

`agent_ppo` 有 3D 进度特征：

- step_norm。
- survival_ratio。
- finished_steps_norm。

`agent_diy` 只有 2D：

- step_norm。
- survival_ratio。

`agent_ppo` 把完成步数也显式输入，`agent_diy` 则更简化。

#### 4.7 奖励设计差异是核心差异

`agent_ppo` 的奖励逻辑更像“潜势函数 + 收集奖励 + 生存奖励”的组合：

- 基础存活奖励 `survive_reward = 0.01`。
- 与怪物距离变化相关的 shaping。
- 宝箱和 buff 收集奖励。
- 物件势能 shaping，使用 `organ_potential` 递推。

它的特点是更偏理论一致性，特别是势能 shaping 那一套，可以尽量减少 pickup 时的突变奖励。

`agent_diy` 的奖励逻辑更直接：

- 基础生存奖励。
- 怪物距离变化奖励。
- 宝箱收集奖励。
- buff 收集奖励。
- 闪现逃脱奖励。
- 远离怪物的方向奖励。

它的特点是更容易把“什么行为有利”直接告诉模型，学习信号更直观。

#### 4.8 奖励尺度也不同

`agent_ppo` 的 reward 尺度更保守，基础奖励是 `0.01`，收集和 shaping 都是小尺度渐进式设计。

`agent_diy` 的 reward 尺度明显更大：

- 生存奖励 `0.15`。
- 宝箱奖励 `10.0`。
- buff 奖励 `2.0`。
- 闪现逃脱奖励 `1.0`。
- 距离 shaping 系数 `0.15`。
- 方向奖励 `0.05`。

这说明 `agent_diy` 的训练信号更“硬”，更容易让模型迅速区分高价值事件。

### 2. [algorithm.py](agent_diy/algorithm/algorithm.py#L28) vs [algorithm.py](agent_ppo/algorithm/algorithm.py#L28)

1. `agent_diy` 的合法动作 softmax 更偏 torch 原生。
   - [agent_diy/_masked_softmax](agent_diy/algorithm/algorithm.py#L160) 直接在 logits 上加非法动作 mask，再使用 `torch.softmax`。
   - [agent_ppo/_masked_softmax](agent_ppo/algorithm/algorithm.py#L149) 先做 `logits * legal_action`，再减最大值并 softmax。

### 3. [model.py](agent_diy/model/model.py#L30) vs [model.py](agent_ppo/model/model.py#L31)

1. `agent_diy` 的 backbone 更深更宽。
   - 结构是 256 → 128 → 64。
   - 见 [agent_diy/model/model.py](agent_diy/model/model.py#L43)。

2. `agent_ppo` 的 backbone 更轻量。
   - 结构是 128 → 64。
   - 见 [agent_ppo/model/model.py](agent_ppo/model/model.py#L39)。

### 4. [conf/train_env_conf.toml](agent_diy/conf/train_env_conf.toml#L1) vs [conf/train_env_conf.toml](agent_ppo/conf/train_env_conf.toml#L1)

1. 地图抽样方式不同。
   - `agent_diy`：`map_random = true`。
   - `agent_ppo`：`map_random = false`。

2. 第二怪出现时机不同。
   - `agent_diy`：`monster_interval = -1`，表示随机。
   - `agent_ppo`：`monster_interval = 300`，固定时机。

3. 怪物加速时机不同。
   - `agent_diy`：`monster_speedup = -1`，表示随机。
   - `agent_ppo`：`monster_speedup = 500`，固定时机。

## 详细差异

### 1. [agent.py](agent_diy/agent.py#L28) vs [agent.py](agent_ppo/agent.py#L28)

这两个文件都负责把环境观测转换成模型输入，执行推理，完成动作采样，并把模型参数保存或加载到磁盘。主流程结构相同，但 `agent_diy` 在工程健壮性和推理细节上更强。

#### 相同点

两者都在初始化时完成以下操作：

- 固定随机种子。
- 创建 `Model`。
- 使用 Adam 优化器。
- 创建 `Algorithm`。
- 创建 `Preprocessor`。
- 维护 `last_action`。

训练时，两者都通过 `predict` 先得到动作概率，再通过 `action_process` 转成最终动作；评估时都通过 `exploit` 使用贪心动作。

#### 差异点

1. `agent_diy` 的加载模型更稳健。
   - 在 [load_model](agent_diy/agent.py#L120) 中会先检查 checkpoint 文件是否存在，不存在时只记录 warning 并跳过。
   - `agent_ppo` 的 [load_model](agent_ppo/agent.py#L115) 直接读取文件，前提是假设 checkpoint 一定存在。

2. `agent_diy` 的推理路径更统一。
   - [agent_diy/_run_model](agent_diy/agent.py#L137) 中先把特征和合法动作转为 tensor，再在 torch 内部完成 masked softmax。
   - [agent_ppo/_run_model](agent_ppo/agent.py#L120) 先得到 logits，再转成 numpy 做合法动作 softmax，属于更手工化的实现。

3. `agent_diy` 的动作处理更明确。
   - [action_process](agent_diy/agent.py#L180) 会根据 `is_stochastic` 明确选择 `action` 或 `d_action`，并统一转成 `int`。
   - `agent_ppo` 的 [action_process](agent_ppo/agent.py#L98) 逻辑更短，直接从 `ActData` 中解包。

4. `agent_diy` 的 `learn` 更防御式。
   - [learn](agent_diy/agent.py#L99) 里如果 `list_sample_data` 为空会直接返回。
   - `agent_ppo` 的 [learn](agent_ppo/agent.py#L98) 没有这个空集保护，默认上游一定会喂入数据。

#### 实际影响

`agent_diy` 在运行时更不容易因为缺 checkpoint 或异常输入而报错，适合更长时间训练和更复杂的实验流程。`agent_ppo` 更轻，但对环境和训练管线的前提要求更强。

---

### 2. [algorithm.py](agent_diy/algorithm/algorithm.py#L28) vs [algorithm.py](agent_ppo/algorithm/algorithm.py#L28)

这两个文件的 PPO 核心损失公式是同一类实现，差别不在“是不是 PPO”，而在实现风格、数值处理和命名习惯。

#### 相同点

两者都包含以下元素：

- 策略损失采用 PPO clipped surrogate objective。
- 价值损失采用 clipped value loss。
- 熵损失用于鼓励探索。
- 最终总损失都形如 `vf_coef * value_loss + policy_loss - beta * entropy_loss`。
- 都会做梯度裁剪。

#### 差异点

1. 变量命名不同，但语义一致。
   - `agent_diy` 用 `self.action_num`、`self.value_num`。
   - `agent_ppo` 用 `self.label_size`、`self.value_num`。
   - 这只是命名差异，不是算法差异。

2. `agent_diy` 的合法动作 softmax 更偏 torch 原生。
   - [agent_diy/_masked_softmax](agent_diy/algorithm/algorithm.py#L160) 直接在 logits 上加非法动作 mask，再使用 `torch.softmax`。
   - [agent_ppo/_masked_softmax](agent_ppo/algorithm/algorithm.py#L149) 先做 `logits * legal_action`，再减最大值并 softmax。

3. `agent_diy` 在 `learn` 中对空数据更安全。
   - [learn](agent_diy/algorithm/algorithm.py#L46) 先检查 `list_sample_data` 是否为空。
   - `agent_ppo` 没有这层保护。

4. 日志和监控写入风格不同。
   - `agent_diy` 在上报监控前会判断 `self.logger` 是否存在。
   - `agent_ppo` 的日志调用更直接，默认 logger 存在。

#### 实际影响

两者的 PPO 公式本身没有本质分歧，因此“训练成不成”不取决于这个文件是否写得花哨。更重要的是前端输入和奖励信号是否足够干净。`agent_diy` 在这个文件上主要是做了更稳健的实现，而不是改 PPO 核心。

---

### 3. [feature/definition.py](agent_diy/feature/definition.py#L1) vs [feature/definition.py](agent_ppo/feature/definition.py#L1)

这个文件定义了观测、动作和样本结构，以及 GAE 计算逻辑。两边核心非常接近，但 `agent_diy` 更像是为扩展奖励接口预留了位置。

#### 相同点

- 都通过 `create_cls` 动态定义 `ObsData`、`ActData`、`SampleData`。
- 都有 `sample_process`，用于填充 `next_value` 并计算 GAE。
- 都用相同的 GAE 公式：`delta = reward - value + gamma * next_value`，然后递推得到 `advantage` 和 `reward_sum`。

#### 差异点

1. `agent_diy` 多了一个 `reward_shaping` 占位函数。
   - [reward_shaping](agent_diy/feature/definition.py#L46) 目前只是 `pass`，说明它预留了对框架奖励接口的兼容点。
   - `agent_ppo` 没有这个函数。

2. `agent_diy` 的注释更偏向说明“这是增强版”。
   - 它明确写了增强特征、增强算法、增强数据类。
   - `agent_ppo` 的描述更接近标准 PPO 实现。

3. 两边 `SampleData` 字段顺序略有不同。
   - 这不影响功能，但反映出两个版本的组织习惯不同。

#### 实际影响

这个文件对性能的直接影响不大，因为它主要是结构定义。真正的差异在于：`agent_diy` 看起来预留了更完整的奖励接入方式，但当前主要行为还是由预处理器里的 reward 逻辑决定。

---

### 4. [feature/preprocessor.py](agent_diy/feature/preprocessor.py#L51) vs [feature/preprocessor.py](agent_ppo/feature/preprocessor.py#L69)

这是最关键的差异文件。两者不仅特征维度不同，奖励策略也不同，基本决定了模型看到什么、学什么、怎么学。

#### 4.1 特征维度和组成不同

`agent_ppo` 的特征总长是 546 维，组成如下：

- 英雄自身 4D。
- 怪物 5D × 2。
- 物件 72D。
- 局部地图 441D。
- 合法动作 16D。
- 进度 3D。

对应配置在 [agent_ppo/conf/conf.py](agent_ppo/conf/conf.py#L17)。

`agent_diy` 的特征总长是 107 维，组成如下：

- 英雄自身 6D。
- 怪物 7D × 2。
- 宝箱 12D。
- buff 8D。
- 局部地图 49D。
- 合法动作 16D。
- 进度 2D。

对应配置在 [agent_diy/conf/conf.py](agent_diy/conf/conf.py#L31)。

#### 4.2 英雄特征不同

`agent_ppo` 只保留：

- 位置 x、z。
- 闪现冷却。
- buff 剩余时间。

`agent_diy` 额外加入：

- buff 是否激活。
- 当前速度归一化。

这意味着 `agent_diy` 对英雄当前状态的刻画更完整，尤其能区分“buff 存在但未激活”和“buff 正在生效”这类细节。

#### 4.3 怪物特征不同

`agent_ppo` 的单个怪物特征是 5D：

- 是否可见。
- 位置 x、z。
- 速度。
- 距离。

`agent_diy` 的单个怪物特征是 7D：

- 是否可见。
- 位置 x、z。
- 速度。
- 距离。
- 相对方向。
- 是否威胁。

也就是说，`agent_diy` 不只告诉模型“怪物在哪”，还告诉模型“这个怪物是不是迫在眉睫”，这会让策略更容易学到避险优先级。

#### 4.4 物件建模方式完全不同

`agent_ppo` 使用一个统一的 72D 物件槽位，编码的是通用 organ 信息：

- sub_type。
- status。
- x。
- z。
- 距离。
- 相对方向。

并且它还会对 organs 按 `config_id` 排序，以减少跨步抖动。

`agent_diy` 则把物件拆成了两类：

- 宝箱 12D，最多 3 个，每个 4 维。
- buff 8D，最多 2 个，每个 4 维。

这种做法的优点是语义更清晰。模型一眼就知道哪些槽位是宝箱，哪些是 buff，不需要自己从通用 organ 向量里再分类型。

#### 4.5 局部地图范围不同

`agent_ppo` 用 21 × 21 的局部地图窗口，共 441D。

`agent_diy` 只用 7 × 7 的中心补丁，共 49D。

这代表两种不同取舍：

- `agent_ppo` 更重视空间上下文和更大范围的障碍布局。
- `agent_diy` 更强调决策中心区域，减少输入噪声。

#### 4.6 进度特征不同

`agent_ppo` 有 3D 进度特征：

- step_norm。
- survival_ratio。
- finished_steps_norm。

`agent_diy` 只有 2D：

- step_norm。
- survival_ratio。

`agent_ppo` 把完成步数也显式输入，`agent_diy` 则更简化。

#### 4.7 奖励设计差异是核心差异

`agent_ppo` 的奖励逻辑更像“潜势函数 + 收集奖励 + 生存奖励”的组合：

- 基础存活奖励 `survive_reward = 0.01`。
- 与怪物距离变化相关的 shaping。
- 宝箱和 buff 收集奖励。
- 物件势能 shaping，使用 `organ_potential` 递推。

它的特点是更偏理论一致性，特别是势能 shaping 那一套，可以尽量减少 pickup 时的突变奖励。

`agent_diy` 的奖励逻辑更直接：

- 基础生存奖励。
- 怪物距离变化奖励。
- 宝箱收集奖励。
- buff 收集奖励。
- 闪现逃脱奖励。
- 远离怪物的方向奖励。

它的特点是更容易把“什么行为有利”直接告诉模型，学习信号更直观。

#### 4.8 奖励尺度也不同

`agent_ppo` 的 reward 尺度更保守，基础奖励是 `0.01`，收集和 shaping 都是小尺度渐进式设计。

`agent_diy` 的 reward 尺度明显更大：

- 生存奖励 `0.15`。
- 宝箱奖励 `10.0`。
- buff 奖励 `2.0`。
- 闪现逃脱奖励 `1.0`。
- 距离 shaping 系数 `0.15`。
- 方向奖励 `0.05`。

这说明 `agent_diy` 的训练信号更“硬”，更容易让模型迅速区分高价值事件。

#### 实际影响

如果只看这个文件，`agent_diy` 之所以可能更强，最关键不是“特征更多”，而是“特征更对齐任务目标，奖励更集中”。`agent_ppo` 的设计更通用、更完整，但也更容易把注意力分散在大量空间信息和复杂潜势上。

---

### 5. [model.py](agent_diy/model/model.py#L30) vs [model.py](agent_ppo/model/model.py#L31)

模型结构的差异比较直接：`agent_diy` 更大，`agent_ppo` 更小。

#### 相同点

- 都是共享 backbone + actor head + critic head。
- 都使用正交初始化。
- 都输出 logits 和 value。
- 都提供 `set_train_mode` 和 `set_eval_mode`。

#### 差异点

1. `agent_diy` 的 backbone 更深更宽。
   - 结构是 256 → 128 → 64。
   - 见 [agent_diy/model/model.py](agent_diy/model/model.py#L43)。

2. `agent_ppo` 的 backbone 更轻量。
   - 结构是 128 → 64。
   - 见 [agent_ppo/model/model.py](agent_ppo/model/model.py#L39)。

3. 模型命名不同。
   - `agent_diy` 使用 `gorge_chase_enhanced_diy`。
   - `agent_ppo` 使用 `gorge_chase_lite`。

#### 实际影响

`agent_diy` 的网络容量更高，适合接更强的特征工程结果；`agent_ppo` 更轻，训练速度和推理开销更低，但表达能力也更有限。

---

### 6. [workflow/train_workflow.py](agent_diy/workflow/train_workflow.py#L23) vs [workflow/train_workflow.py](agent_ppo/workflow/train_workflow.py#L23)

训练工作流基本一致，说明两者的训练管道框架没有分叉，差异主要在配置文件路径和少量健壮性处理。

#### 相同点

- 都读取用户环境配置。
- 都创建 `EpisodeRunner`。
- 都循环调用 `run_episodes`。
- 都在固定周期保存模型。
- 都使用 `sample_process` 做 GAE 处理。

#### 差异点

1. 配置文件路径不同。
   - `agent_diy` 读取 [agent_diy/conf/train_env_conf.toml](agent_diy/conf/train_env_conf.toml#L1)。
   - `agent_ppo` 读取 [agent_ppo/conf/train_env_conf.toml](agent_ppo/conf/train_env_conf.toml#L1)。

2. 日志保护程度略有不同。
   - `agent_diy` 在打印训练指标时会更注意 `logger` 是否存在。
   - `agent_ppo` 的日志调用更直接。

3. 终局日志和监控上报逻辑几乎相同，但 `agent_diy` 的工程注释更强调“增强版”的意图。

#### 实际影响

工作流层面差异不大，说明两者都沿用了同一训练框架。真正决定结果的仍然是前面的特征、奖励、模型和环境配置。

---

### 7. [conf/conf.py](agent_diy/conf/conf.py#L18) vs [conf/conf.py](agent_ppo/conf/conf.py#L14)

这个文件直接反映两个版本的设计目标。

#### 相同点

两者都定义了：

- `ACTION_NUM = 16`。
- `VALUE_NUM = 1`。
- `GAMMA = 0.99`。
- `LAMDA = 0.95`。
- `INIT_LEARNING_RATE_START = 0.0003`。
- `BETA_START = 0.001`。
- `CLIP_PARAM = 0.2`。
- `VF_COEF = 1.0`。
- `GRAD_CLIP_RANGE = 0.5`。

#### 差异点

1. `agent_ppo` 的配置更简洁。
   - 只保留了特征长度和 PPO 的核心超参数。

2. `agent_diy` 的配置更完整。
   - 明确列出特征拆分 [agent_diy/conf/conf.py](agent_diy/conf/conf.py#L31)。
   - 明确列出网络结构 [agent_diy/conf/conf.py](agent_diy/conf/conf.py#L73)。
   - 明确列出奖励超参数 [agent_diy/conf/conf.py](agent_diy/conf/conf.py#L89)。

3. `agent_diy` 使用 107 维输入定义。
   - `FEATURE_DIMS = [6, 7, 7, 12, 8, 49, 16, 2]`。
   - 这与其预处理器一致。

4. `agent_ppo` 使用 546 维输入定义。
   - `FEATURES = [4, 5, 5, 72, 441, 16, 3]`。
   - 这与其预处理器一致。

#### 实际影响

`agent_diy` 的配置更像一个可以继续调参的实验底座；`agent_ppo` 更像一个固定基线配置。前者更适合继续迭代，后者更适合做稳定对照实验。

---

### 8. [conf/monitor_builder.py](agent_diy/conf/monitor_builder.py#L14) vs [conf/monitor_builder.py](agent_ppo/conf/monitor_builder.py#L17)

这两个文件基本一致，说明监控面板本身没有分叉。

#### 相同点

- 都使用 `MonitorConfigBuilder`。
- 都构建了同样的指标：reward、total_loss、value_loss、policy_loss、entropy_loss。
- 都放在同一个“算法指标”分组里。

#### 差异点

几乎没有本质差异，更多是注释和文案上的微调。

#### 实际影响

监控层面你看到的曲线是可比的，这一点很好，因为它意味着二者的性能差异不是“看板定义不同”造成的，而是算法和输入本身造成的。

---

### 9. [conf/train_env_conf.toml](agent_diy/conf/train_env_conf.toml#L1) vs [conf/train_env_conf.toml](agent_ppo/conf/train_env_conf.toml#L1)

环境配置是两者训练分布差异最直观的来源之一。

#### 相同点

- `map` 都是 1 到 10。
- `treasure_count = 10`。
- `buff_count = 2`。
- `buff_cooldown = 200`。
- `talent_cooldown = 100`。
- `max_step = 1000`。

#### 差异点

1. 地图抽样方式不同。
   - `agent_diy`：`map_random = true`。
   - `agent_ppo`：`map_random = false`。

2. 第二怪出现时机不同。
   - `agent_diy`：`monster_interval = -1`，表示随机。
   - `agent_ppo`：`monster_interval = 300`，固定时机。

3. 怪物加速时机不同。
   - `agent_diy`：`monster_speedup = -1`，表示随机。
   - `agent_ppo`：`monster_speedup = 500`，固定时机。

#### 实际影响

`agent_ppo` 的环境节奏更固定，训练更稳定，也更容易复现实验结果。`agent_diy` 的随机性更强，训练分布更丰富，泛化潜力通常更高，但也更难训。

---

## 其他文件

以下文件在两边基本没有实质差异，通常只是空包或导出文件：

- `__init__.py`
- `algorithm/__init__.py`
- `feature/__init__.py`
- `model/__init__.py`
- `workflow/__init__.py`
- `conf/__init__.py`

这些文件不影响核心行为，因此不构成有效差异。

## 你可以怎么理解这两个版本

如果把两套代码抽象成一句话：

- `agent_ppo` 是“更标准、更直接、更像基线”的版本。
- `agent_diy` 是“更聚焦、更工程化、更强调任务对齐”的版本。

从设计哲学上看，`agent_ppo` 倾向于保留更多原始环境信息，再让模型自己去学；`agent_diy` 倾向于先把人类认为重要的信息整理出来，再让模型在更干净的输入上学习。

这也是为什么 `agent_diy` 可能更容易取得更好的效果：

- 输入更少但更有针对性。
- 奖励更明确，减少了无效学习信号。
- 模型更大，能承载更复杂的策略映射。
- 工程上更稳健，训练过程中更不容易因为边界问题出错。

## 简短结论

如果只保留最核心的一句话：`agent_ppo` 的优势在于结构简洁和可复现，`agent_diy` 的优势在于更强的特征压缩、更直接的奖励塑形、更大的模型容量和更好的训练信号对齐。真正让二者拉开差距的，不是 PPO 框架本身，而是输入、奖励和环境随机性的设计。