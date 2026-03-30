ENOTO: Improving o2o RL with Q-ensembles
**重点关注实验**
# 1. Introduction
## Abstract
ENOTO增加Q-networks的数量，使得从offline pre-training到online FT之间的性能不会下降。
为了加速online performance enhancement的速度，我们loosen pessimism of Q-Estimation
## Intro
我们尝试添加Q-networks的数量，发现Q-Ensembles可以帮助减轻unstable training与performance degradation的问题。
我们可以消除原本的Pessimistic Term。
主要的创新
- 展现offline$\rightarrow$online setting中Q-ensemble对于unstable training与performance drop问题的解决能力。
- ENOTO是一个unified framework，可以使用许多offline RL algorithms
- 经验上分析ENOTO在benchmark上的表现，证明其SOTA性能
# 2.Why can Q-Ensembles help o2o
![[Pasted image 20250828145445.png]]
如果直接offline->online，pessimism会直接影响下一步的训练
我们进行实验
- 如果自始至终使用CQL(offline, online)，可以缓慢上升，但是一直保持pessimistic，在online阶段缺乏了探索性
- 如果直接转换为Online algorithm，则会导致不稳定的结果，或者性能下滑。这是因为SAC没有对Q值的准确估计。
- Q-Ensembles将Q-functions的数量设计为N，我们再选择最小的那个Q作为我们的Q-Target
不难发现CQL-N->SAC-N这种方法最终的效果稳定且得分较高。
我们发现training非常稳定，且performance drop也没有再被观测到。
**不仅offline训练结果更好了**，online训练之后的结果也更好了。
如果直接移除Pessimistic Constraints，Q-value会剧烈波动，导致训练不稳定。但是我们的Q-Ensembles中，在online阶段SAC-N仍然能够有Conservative的估计。
- 得出结论：保守性是非常重要的。
SAC-N与CQL均能够在online fine-tuning中防止performance drop，但是为什么SAC-N的性能更好？
- SAC-N的action choices范围比CQL更广，可以有更好的性能。
- Q-Ensembles可以保证保守性质，而且比CQL更有探索性。
# 3. Ensemble-based o2o RL
介绍Ensemble-Based o2o Framework
## 3.1 Q-Ensembles
*OfflineRL-N*在*OfflineRL*的基础上的改动就是使用N个独立的Q-networks，从所有Q值中挑选最小的用来计算target
我们先预训练一个*OfflineRL-N* agent，将其作为online agent的初始化，移除pessimistic term，例如从CQL-N到SAC-N
我们在MuJoCo环境中进行实验
![[Pasted image 20250828152455.png]]
我们在这里可以看出OfflineRL-N -> OnlineRL-N的结果是最为理想的。
- offlineRL->offlineRL这种解决方案虽然稳定，但是渐进性能较差
- 直接转化为OnlineRL会导致不稳定的训练过程，以及Performance Drop
- Q-Ensembles这种解决方案能够兼顾稳定性与较好的性能。
虽然这种Ensemble-Based的方法比传统的offline->offline好一些，但是改进的速度仍然比不上传统的online RL algorithms
## 3.2 Loosing Pessimism
在前一个section中，我们将OnlineRL-N当作是我们的online阶段的算法，同样需要选择最小的Q-value俩计算target
但是这样的操作似乎还是**过于保守了**
我们需要找到一个新方法，能够loosen pessimistic estimation，同时也保持stable的特性。
我们在这里提出几种Q-target computation methods
- MinQ：就是选择所有Q值中最小的来计算target
- MeanQ：计算所有Q-values的平均值
- REM：一种原本用来增强DQN性能的方法，使用随机凸组合，所有权重不能为负，且权重总量归一化（一种加权方式），同样是一种mean，但是相较于MeanQ有更多的随机性。
- RandomMinPair：随机挑选出两个Q-functions，并取这两个Q-functions中的最小值。
- WeightedMinPair：计算所有RandomMinPair Targets的期望，类似于WeightedMinPair的随机抽样版本，可以降低随机性。
MeanQ与REM的性能都非常垃圾，但是RandomMinPair与WeightedMinPair表现较好。
可以发现WeightedMinPair的稳定性会更好。
所以之后我们默认的算法就是OnlineRL-N+WeightedMinPair
![[Pasted image 20250828160156.png]]
## 3.3 Optimistic Exploration
前面我们使用Pessimistic Learning获得了Online Learning一个令人满意的开头，且减弱了Online Learning中的Conservatism.
这一章使用Ensemble-Based Exploration方法来进一步优化我们的性能。
接下来我们对比三种exploration methods
1. Bootstrapped DQN
	 - 创建N个Bootstrapped heads独自进行branching
	 - 
2. OAC
	- off-policy exploration strategy
	- 最大化upper confidence bound
	- 不确定性下的乐观原则，假设高度不确定性的行为有更高的奖励。
3. SUNRISE
	- 提供Ensemble-based weighted Bellman backups
	- 对于不确定性作为权重进行加权
	- 高不确定性的Q值权重更高，可以引导我们的更新方向。
最终的结果是OnlineRL-N+WeightedMinPair+SUNRISE的综合结果是最好的，所以这就是我们的ENOTO Framework
### Pseudo-Code
![[Pasted image 20250828163619.png]]
Offline阶段
- OfflineRL-N集成Q-ensembles算法进行训练
- 使用的数据集是离线数据集$D_{offline}$
Online阶段
- 移除Pessimistic Term，从OfflineRL-N转化为OnlineRL-N
- Q-Target计算方式使用WeightedMinPair
- 使用SUNRISE方法进行exploration
# 4. Experiments
## 4.1 Locomotion Tasks
我们使用三种dataset types
- medium：中等级别的policy采样得到的结果
- medium-replay：从头训练一个medium-level agent过程中遇到的所有数据。
- medium-expert：混合了medium-level与expert-level的数据的样本
### Comparative Evaluation
我们的baseline如下：
- AWAC：我们更加重点学习high advantage的那些action
- BR: 一种o2o RL算法，训练额外的network来调整sample的优先级
- PEX：policy set中同时有offline policy与online policy，根据其Q值通过softmax函数进行采样
- Cal-QL：校准Q值防止过度Underestimation
- IQL：一种offline RL算法，且online FT效果也很理想
- SAC：一种Online RL算法
- Scratch：SAC-N+WMP+SUNRISE直接进行online training
![[Pasted image 20250829143603.png]]
下面我们**根据实验分组**分别阐述我们算法的优势。
- 对比pure online RL：初始性能良好，证明Offline Pre-training的作用
- 对比offline RL(IQL)，我们的方法的fine-tuning速度更快，因为没有pessimism的training更适合online fine-tuning的要求。
- 对比其他o2o RL：
	- AWAC由数据集的质量所限制
	- BR的性能仅次于ENOTO-CQL，但是训练不稳定
	- PEX在online FT初始阶段有性能下滑问题
	- Cal-QL的稳定性是非常好的
## 4.2 Navigation Tasks
这里使用Antmaze作为我们进行比较的项目
![[Pasted image 20250829145653.png]]
首先，LAPO比IQL的离线性能更好，所以会有更高的起点
其次，IQL的asymptotic performance会因为offline constraints而更低，PEX会在一定程度上增强exploration的强度
PEX的performance drop会更加严重
综合来看，我们的ENOTO-LAPO方法不仅在offline stage的性能令人满意，且在online阶段同样可以维持较好的性能，不会有严重的Degradation。
# 5. Conclusions and Limitations
这里展现了Q-Ensembles对于缓解Unstable Training与Performance Drop是非常有帮助的，是一种更为**灵活的Constraint Method**
