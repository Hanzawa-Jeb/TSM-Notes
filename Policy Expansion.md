# Policy Expansion (PEX) - Bridging Offline-to-Online RL

## 背景
- **问题**: 直接用 offline 训练好的 actor 做 online finetune，常常导致性能下降，甚至遗忘已有的好行为。
- **原因**: 
  - critic 冷启动，初期梯度噪声大；
  - offline policy 太保守，不利于在线探索；
  - distribution shift 导致学习不稳定。

## 核心思想
- **Policy Expansion (PEX)**:
  - 离线阶段训练得到策略 **πβ (offline policy)** 和 critic **Qφ**；
  - 在线阶段不直接 finetune πβ，而是 **冻结 πβ**，并新建一个可学习的策略 **πθ (online policy)**；
  - 构建策略集合：
    $$
    \Pi = [\pi_\beta, \pi_\theta]
    $$

- **Adaptive Policy Composition**:
  - 在状态 s 下，每个策略提出一个候选动作：
    $$
    A = \{a_i \sim \pi_i(s) \mid \pi_i \in \Pi\}
    $$
  - 用 critic Qφ(s,a) 评估动作价值，并构建 softmax 分布：
    $$
    P_w[i] = \frac{\exp(Q_\phi(s,a_i)/\alpha)}{\sum_j \exp(Q_\phi(s,a_j)/\alpha)}
    $$
  - 按概率 P_w 选择最终执行的动作。
  - 合成策略可写为：
    $$
    \tilde{\pi}(a|s) = [\delta_{a \sim \pi_\beta(s)}, \delta_{a \sim \pi_\theta(s)}]_w
    $$
    - 其中 δ 表示 Dirac delta，表示确定性选择某个策略输出的动作。

## Critic 的使用
- **离线阶段**：训练 critic Qφ，用于价值评估。
- **在线阶段**：继续训练同一个 Qφ，使用离线数据 + 在线数据混合更新。
- Critic 在整个过程中是“共享”的。
- ==非常关键！Critic是直接从Offline Stage继承而来的！==，我们**不需要重新冷启动**一个全新的Critic

## Actor 的更新
- **πβ**: 冻结，不再更新。
- **πθ**: 在线阶段更新，loss 来自 IQL 风格的 advantage-weighted behavior cloning：
  $$
  L_{\pi_\theta} = -\mathbb{E}_{(s,a) \sim D_{\text{offline}} \cup D} [ w(s,a) \cdot \log \pi_\theta(a|s) ]
  $$
- 因此，**无论动作来自 offline actor 还是 online actor**，都会被用来更新 πθ。
- IQL风格的AWBC方法的特点就是无论是不是自己的行为，都可以被用来更新。
- 这种IQL风格的AWBC方法，**可以同时使用offline与online数据集来训练自己的行为。**

## 优势
1. **保留离线策略**：不会破坏 offline actor 已学到的行为。
2. **灵活性**：offline actor 和 online actor 可以有不同结构。
3. **自适应组合**：状态相关地选择 offline/online 策略，充分利用二者优势。
4. **平滑衔接**：critic 的连续使用避免了 online 阶段冷启动问题。

## 与 DAgger 的关系
- 相似点：都涉及 “已有策略 + 新策略” 的组合。
- 区别：
  - **DAgger**：模仿学习场景，组合权重是预设/固定调度。
  - **PEX**：offline-to-online RL 场景，组合权重是 state-dependent（由 Q 值决定）。
  - DAgger也是一种混合两种策略的方法，混合了原有的policy与新的policy，但是这个占比在DAgger中是事先固定的。