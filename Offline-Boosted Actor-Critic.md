- 很遗憾，前面的笔记因为电脑损坏已经丢失。下次一定要记得用git多多保存呀。
# 1. Introduction
## Abstract
当前的off-policy方法仍然无法以较高的效率利用数据
在这篇工作中，我们在shared online replay buffer上训练offline RL policy，回避原始的online learning policy有更好的表现。
我们使用offline optimal policy来优化online policy learning
我们在这篇工作中提出OBAC这种方法
- online-RL Framework
- 找到表现好的offline policy（通过Value Comparison）
- 我们用这个当做一个adaptive constraint来保证更好的policy learning performance
- 性能较为优良。
## Introduction
### Brief Info
online model-free RL算法在一系列的sequential decision-making tasks中都有良好的表现。
很多优点都是来自于off-policy RL methods的一系列优势（因为这些方法**可以利用historical policies**）
但是这样需要millions of environment interaction steps的方法仍然会防止real-world deployment of RL
我们需要完成的步骤：
- Policy Evaluation for **value function learning**
- Policy Extraction via value maximization
大多数的算法都没有利用Replay Buffer中的数据分布差异，而是将其看作是**一个完整的Data Pool**，而没有利用到Inherent Patterns and knowledge from the heterogeneous data的优势。
### Solutions
有一种解决方案是Model-Based RL，使用数据学习当前的环境Dynamics Model，生成全新的pseudo-sample
- 但是这种方法计算量大，且较为敏感。
还有一种解决方案是offline RL，从fixed datasets中进行学习。
- offline-learned policy可以视作一个explicit performance baseline
- pessimistic training scheme确保在分布内的数据的准确估计。
现在我们移植到online off-policy RL training之中，
- 加入一个additional offline dataset for sampling augmentation，也就是使用额外的一个预先手机号的数据集。但是这个方法成本高
- 先学习一个optimal offline value function，来调整online value function，也就是利用离线学习到的最优值函数来调整online learning的值函数。这种方法可能会导致不准确的Policy Evaluation
### OBAC
我们要解决什么时候使用online policy，什么时候使用offline policy的问题。
我们需要形成正向循环，在离线策略更优的时候，会引导在线策略更好探索，收集数据质量更高。
- OBAC算法可以使用offline optimal policy来对policy进行constraint
- 我们可以在offlien policy更好的时候逐渐增进我们的性能，形成一个正循环，更好的采样能够更好训练online policy
- 为了规避计算复杂度，我们介绍了implicit offline policy learning同样在evaluation与improvement step进行学习。
且计算效率也会更高。
最后的结果令人满意。
# 2. Preliminaries
我们的Objective就是最大化一个轨迹上的r值的discounted sum
我们使用的是off-policy RL，我们将新采集的数据录入Replay Buffer
我们的offline optimal policy就应该是在当前state能够获得最大的action value对应的那个action
**特殊之处在于**，我们同时训练一个online learning policy与offline optimal policy(公用同一个replay buffer)
- $\pi_k$是online policy，$\mu^*_k$是offline optimal policy
- 这样的方法可以保证，offline policy给出的action往往是restricted的，而online policy给出的action是unrestricted的
- 所以offline policy可以认为是historical给出的performance baseline
# 3. Offline-Boosted Off-Policy RL
我们介绍OBAC
- 核心在于 使用offline optimal policy来优化我们的online learning policy
- 将offline optimal policy集成进policy evaluation中
## 3.1 A Motivating Example
offline RL一般都在policy training中遵循pessimism的基本原则来防止extrapolation error
在online RL中，因为new samples是不断在interaction中被采样的，replay buffer不断变化，所以不再需要pessimism
我们需要三个不同的Agent
- Pure off-policy agent：
	- SAC agent，每一步的数据都被加入buffer
	- 就是最普通的off-policy算法
- Concurrent offline：
	- IQL学习离线，同时动态更改SAC buffer，IQL agent不与环境交互
	- SAC作为交互算法，SAC的buffer同时也作为IQL的训练池
	- IQL使用SAC的采样buffer同时进行训练
- Online training offline agent：
	- IQL agent与环境交互，使用buffer中的数据直接进行更新。
	- IQL在自己采样的buffer中进行更新
	- 也就是offline algo直接用于online set
![[Pasted image 20250828114003.png]]
- 多数情况下都是IQL Concurrent有最好的训练效果。
- 我们可以发现offline optimal policy可能会比online的效果更好，即使使用的是相同的replay buffer
- 但是我们可以发现如果没有online buffer，会展现出Q-value的严重低估
这里==有一个重点需要注意：==
- IQL online虽然buffer是online改变的，但是仍然具有offline RL的保守特性
- IQL online容易出现过早收敛的问题。
但是有问题：何时offline optimal policy会比online更好是一个不确定的问题，依赖于online interaction的质量。
## 3.2 Derivation of Offline-Boosted Policy Iteration
数学推导，可以后面再读
所以我们执行的策略始终是online policy，只是在offline policy更强的时候将其作为约束
## 3.3 Offline-Boosted Actor-Critic
我们的Q值与V值函数中同样考虑了offline optimal policy与online learning policy
我们的策略被定义为一个简单的Gaussian Policy
### Policy Evaluation
我们的Q是通过Bellman Expectation Operator得出的
V就是通过Q的期望得出的
我们学习两个Q函数，离线Q的学习方式借鉴了IQL等离线RL算法，高expectile因子使其偏向高回报样本。

### Policy Improvement
![[Pasted image 20250828131333.png]]
在Online policy更好的时候，我们直接进行传统的更新
在offline policy更好的时候，我们融入offline policy作为约束更新。这里是将概率分布直接进行乘积
这个就体现了我们的**Adaptive**这种性质。
原本是SAC-Style的更新方式。
![[Pasted image 20250828134005.png]]
这个就是SAC风格的更新方式。
### Pseudo-Code
![[Pasted image 20250828130508.png]]

# 4. Experiments
## 4.2 Ablation Studies
### Necessity of Adaptive Constraints
我们将OBAC(Adaptive)与fixed constraint OBAC之间进行比较，发现adaptive拥有最好的结果。
### Extension in noise and sparse tasks
在有噪声影响的工作中，我们仍然有较好的效果
# 6. Conclusion
OBAC的主要创新点就在于，我们可以利用一个在Replay Buffer中训练的offline optimal policy来帮助online learning policy。
这是一种全新的结合off-policy RL and offline RL的方法。
**请注意**，我们优化的对象始终都是online policy，offline policy的目的更像是一种指导，可以分析过去的数据并给出建议，分析历史中合适的action与policy