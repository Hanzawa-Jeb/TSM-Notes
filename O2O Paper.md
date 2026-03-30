# Online Pre-Training for o2o RL
icml25
![[Pasted image 20260203123426.png]]
motivation: 可以看出这里from scratch的性能表现提升会更加明显
前面有一些认为这个是inaccurate value estimation导致的问题,CalQL就是着手解决这个问题。
本篇论文提出是否能通过new value function来解决slow performance improvement
提出的方法是OPT, Online Pre-Training for o2o RL
不仅仅依赖于原本offline阶段的value function，而是使用一个新的value function
Offline RL为了防止OOD动作，往往会对value function进行改造，例如加上Conservatism来防止采样OOD动作，但是Online阶段如果仍然conservative就会导致问题
## Methods
Online Pre-Training
![[Pasted image 20260203145302.png]]
这是论文中提出的一整个过程，也就是在online fine-tuning之前固定原来的Critic和Actor,利用Online Buffer训练新的Q
Offline Pre-trainnig: 训练$Q^{off-pt}$和$\pi^{off}$ 
Online Pretraining: 在offline dataset和online samples上训练$Q^{on-pt}$
Online Finetuning: 同时使用$Q^{off-pt}$和$Q^{on-pt}$，两个value function同时会用到

Online Pretraining的时候，训练一个from scratch的online Q模型
为了防止from scratch 训练这个Q-Model可能会对Policy带来影响，引入这个Online Pre-Training Phase

在针对Datasets时，在online pre-training阶段同时引入online samples，通过$\pi^{off}$来在新的环境中进行online采样

Designing Objective Function:
- ![[Pasted image 20260203151836.png]]
- 这里的Objective实际上引入了两个部分的Loss，一个是当前的Q在offline datasets上的Loss，第二个则是考虑了在新的online datasets上的Loss，后面这个部分，是考虑了当前的Q在offline dataset之后更新得到的结果的Loss
- 第一项考虑了从offline dataset中学习到的知识，第二项则考虑了对于$B_{on}$的适应能力
- 注意这里是直接一次性采样完全部需要的数据再进行训练
Online Fine-Tuning阶段
- 不停填充$B_{on}$
- 但是注意在这一阶段，仍然会同时训练$Q^{off}$和$Q^{on}$
- 这一阶段仍然会平衡使用两个Q的估计
![[Pasted image 20260203153937.png]]
- 这里的$\kappa$是一个weighting coefficient，在$\kappa$更小的时候，更大程度依赖offline的。$\kappa$可以进行递增，在晚期更多依赖于online的这个model
Balanced Replay
- 使用了Balanced Replay来平衡online interactions与offline dataset
PseudoCode
![[Pasted image 20260203153035.png]]

可以参考一下他这里对比的这些分数
![[Pasted image 20260203154919.png]]
感觉Antmaze也并不是特别高，能跑到这个分应该不是很难
==注意一下他这里跑这些不同的tasks用的是不一样的算法==
# Penalizing Infeasible Actions and Reward Scaling in Reinforcement Learning with Offline Data
icml25
- o2o RL中因为offline data的覆盖范围有限，可能会导致Q-Function的Extrapolation Error，OOD Actions的value被高估。
- 本质上是因为data range之外的Q值倾向于变成一种线性关系。
- Layernorm可以限制Q-function predictions，因为可以限定范数，但是LN是一种bounding而不是直接reduce，所以使用两种方法来解决这个问题。
- **Reward-Scaling with Layer Normalization**
- **Penalizing Infeasible Actions**
LayerNorm以及Increasing Reward Scale可以降低ID与OOD actions相似度，类内梯度更新就会较少影响OOD Q-Est
- 使用NTK确定梯度更新的相关性，如果ID和OOD的NTK大，那么说明ID会影响OOD的Q-Approximation
- NTK(Neural Tangent Kernel)
Based on TD3
### Extrapolation Error outside the convex hull
![[Pasted image 20260205151120.png]]
- OOD Actions指的就是Feasible Action Set中不属于数据集中的这些Actions
- 这些已经存在的点可以通过加权形成一个Convex Hull
- 通过实验可以证明，发现OOD-out中的这些估计会变得高度线性化，导致OOD-out的估计值出现问题
### Good Extrapolation?
- 我们希望这些OOD-out的值更低，确保optimal actions within the data应该才是我们选择的动作
![[Pasted image 20260205152918.png]]
期望的是这样的
## Penalizing Infeasible Actions and Reward Scaling
### Reward Scaling with LN(RS-LN)
- 我们可以发现训练一个target的时候，其他也会受影响
- 如果Q认为OOD-out的action与分部内的不相似，那么对OOD影响就不明显，就不会导致OOD的值很高(因为初始值并不高)
- 所以目标就是鼓励Q值在dataset内的分布与OOD-out的分布尽量远离
- 实际上就是要促进feature resolution
![[Pasted image 20260205154257.png]]
- 从这张示例图中可以看出，Reducing the rror会需要更加精细的input space，也就是更高的resolution
- 在output scale increase的时候，网络会被倒逼学习更加fine-grained的features
![[Pasted image 20260205155427.png]]
- 可以看出Layernorm可以减缓Overestimation
- creward就是reward scaling factor，可以看出加大reward scaling factor可以降低OOD的值
![[Pasted image 20260205160032.png]]
- 这里是NTK的关系，可以发现在creward更大的时候，相关性会变小
![[Pasted image 20260205160335.png]]
- 这里我们可以注意到这些现象(和我们之前的实验很类似)
## Penalizing Infeasible Actions(PA)
- 这里加上了一个hard constraint，保证trend downward
- 在far from feasible region的地方设置penalties，以及在feasible region之外可以设置guard interval
![[Pasted image 20260205161541.png]]
- 引入了Infeasible Actions的Loss
![[Pasted image 20260205161857.png]]
- 压低infeasible的值，这样可以给整个Landscape做锚定，防止Boundary值的Overestimation
![[Pasted image 20260205162521.png]]
### PARS Algorithm
Infeasible Action Sampling
- 通过Sample Expectation实现
- ![[Pasted image 20260205163108.png]]
Critic Ensemble
- 
# Self-Improving VLA with DataGen via ResidualRL
传统的VLM的作用：Vision+Text -> VLM -> Text
- 通过视觉与文字的输入，生成对应的文字
- VLA中，VLM的输入是视觉与语言，VLM将一个词语与像素点关联，这就是Grounding。输出则是一个指令性的动作Token
- VLA的SFT: SFT继承自NLP领域，是一种Imitation Learning，需要来自于专家数据。
- 训练从vision+text输入 到 expert action的映射
- SFT改变了VLM的原始输出，让输出从token变成Action
## Method
- Generalist VLA是一个在多任务环境上得到训练的通用型策略模型
- 我们是要加上specialist(通过RL或者示范数据跑出来的特定任务)
- 整体的思路是先freeze base policy $\pi_b$ ，学习一个lightweight residual action policy $\pi_\delta$ 。使用这个residual part在base policy probing之后进行进一步探索，最终将这些skills通过SFT蒸馏回Generalist Policy中
### Data Efficient RL via Policy Prior Warm-Start
- 首先在offline buffer中存储successful rollouts
- 从base policy中得来
- 训练task-specific residual action module，使用$\pi_\delta$寻找more optimal solutions，同时缩小了范围，感觉有点像是TRPO的思路，限制了探索的范围
- residual policy是非常容易训练的
- 在warm-up stage，只使用base policy，Q-function使用的是conservative objective
### Bootstrapping RL Specialist for Scalable DataGen
- RL Specialist采样可能会非常optimal, consistent，导致OOD现象，会导致overfitting
- 使用base-policy initialization
- 先试用base-policy进行random steps，接下来让residual RL接手，视作base policy probing，促进了robustness
- probing steps并不会被进入replay buffer
![[Pasted image 20260204141935.png]]

## 总结
- 原本的VLA需要大量的人类示范数据，才能从generalist SFT形成specialist
- 问题是demo昂贵且覆盖不全面
- 所以现在让VLA自己生成高价值的数据
PLD
- 让base policy先暴露出自己会失败的状态，RL Specialist完成任务，蒸馏回原来的backbone networks中
- 通过Base Policy Probing，发现失效区域，再生成Recovery Behaviours的轨迹
- 学习机制从传统的BC损失函数，变为Residual RL
- PLD实际发挥的作用就是一个Data Generator，在SFT阶段使用的仍然是BC，只不过数据都是由PLD生成的
![[Pasted image 20260204144603.png]]

# Network Sparsity Unlocks the Scaling Potential of DRL
icml25 oral
![[Pasted image 20260205125701.png]]
- DRL中的Scaling反而会导致optimization的问题
- 提出了Network Sparsity的重要性，可以达到更好的scaling效果
- sparse networks能够有所帮助，Plasticity Loss这一类的问题会在increasing model scale的时候变得越来越明显
One-Shot Random Pruning
- 建立一个fixed sparse topology
- Random Pruning就是在每一层初始化时设定binary masks，如果是0这一个weight
Sparsity Ratio
- Uniform Sparse: 每一层的稀疏比例都设置为整个网络的稀疏比例
- Erdos-Renyi:根据层的输入和输出维度来动态调整每一层的稀疏度
Sparsity Promotes DRL Network Scaling
- 合适的Network Sparsity可以促进Model Scaling，促进Efficiency
- larger dense networks在参数效率上受到更大的影响
- ![[Pasted image 20260205120642.png]]
- 在参数量相同的情况下，又大又稀疏的网络的性能比又小又稠密的网络要好
- 总结就是Weight-Level Sparsity可以有效促进Scaling Potential of DRL Networks
Interplay between Model Size and Sparsity
- large networks中，更高的sparsity ratios会有更高的性能，尤其是在Humanoid Walk和Dog Run环境中
- 但是在小网络中，不能有太高的Sparsity Ratio，否则就会导致表达能力不足
### 为什么Sparse Networks比Dense Counterparts性能更好
Representational Capacity
- 目的是有enhaned expressivity，捕捉更多复杂的关系
- Stable Rank(Srank)可以衡量有效秩，代表网络学习到的Representations
- Network Sparsity增长-> Srank增长
Plasticity
- 由Dormant Ratio和Gradient Norm来体现
- 可以发现large sparse网络的可塑性指标是最强的
- large dense网络会有非常严重的plasticity deterioration
Reset as a Diagnostic Tool
- Large Dense Networks的Reset会有性能提升，而large sparse网络并没有理想，说明了large-scale networks中sparsity对于维持可塑性的重要作用
Regularization
- Parameter Norm，可以发现Large Dense Network的Norm非常大，而Large Sparse还不错
![[Pasted image 20260205123320.png]]
- 值得注意的是large sparse即使数据量差不多，norm也会更低
Simplicity Bias
- network sparsity的情况也能够具有更高的simplicity bias scores
Gradient Interference:
- 衡量gradient之间的相关性对于训练的影响，也就是Lyle有关Gradient Covariance那一篇的工作
- 可以发现large sparse可以促进梯度正交性，防止gradient interference
### Conclusion
- static network sparsity，可以从多个方面解决optimization pathologies相关的问题，以及提供更好的性能表现
- 注意这里设置为0的是weights
# Group Meeting
Wasserstein距离作为Distillation Penalty???
TV散度？其他散度？

# Uni-RL (nips 25)
IVR的定义：
- 通过soft-限制Q值来达到限制policy的作用，尽量避免Overestimation以及extrapolation error等问题
当前不同的RL Setting面临不同的问题，不同的算法同样适用于不同的环境
Unified RL Framework
- 应该要能适用于online, offline和offline to online环境
- 要有较高的Sample Efficiency，对Env Interactions的要求不高
- Scalability: 数据量增长时也有更好的性能表现
- 下图为Eq.1
![[Pasted image 20260209121902.png]]
- 这一步Policy的要求仍然是reward最大，同时确保和reference policy之间的距离在f-sphere内
- 为了转化为unconstrained, regularized one，添加一个新的regularization penalty(f-divergence)
- offline RL中的reference policy $\mu$ 就是behaviour policy
- online RL中的$\mu$ 就是target policy，也就是那个softly updated to the policy的
- 对于o2o RL而言，offline阶段和online阶段就分别是上面提到过的这两种
==Lagrangian Relaxation==:将硬性Constraint转化为Penalty
### IVR (Value)
- 目的是推导出数学公式的Dual Form，以及为什么对偶形式可以提供相同的IVR效果
- 原式经过拉格朗日松弛生成一个penalty，接下来还可以推导出一个新的Value，这里的Value中是考虑了与原策略之间的差距的，这里的value就已经考虑到了这个penalty
![[Pasted image 20260209125855.png]]
这里是IVR的形式，也就是在Policy Optimization目标的基础上添加了一个对于reward的penalty，这个表达式可以等效于解决一个Behavior-Regularized MDP
![[Pasted image 20260209130036.png]]
使用对偶性的证明，我们可以推导出Q和V的最优解应该满足如下条件
![[Pasted image 20260209130410.png]]
注意这里的表达式只依赖于offline datasets中的知识，不需要知道具体的policy
==IVR的核心创新点==：
- 在value中引入divergence penalty
- 解耦value与policy的计算，value的计算只需要依赖自己即可
使用Duality证明，因为前面的Action是从offline dataset中采样得到的，这里就体现了Base Policy
#### Does IVR work in the online setting
- online IVR中，采集的数据集来自于收集的数据，可能会包含大量的suboptimal data，如果限制于behaviour policy，可能会导致性能上限不高
![[Pasted image 20260209135421.png]]
注意这张图中，如果使用$\pi_D$作为reference policy，容易导致性能上限不高
![[Pasted image 20260209135648.png]]
修改这里的reference policy即可保证一定程度上的更新，如果限制在原来offline dataset中的policy，则会导致性能上限低下
### InPG (Policy Extraction)
![[Pasted image 20260209141038.png]]
第一种方法是weighted BC，使用Forward KL Divergence
![[Pasted image 20260223161557.png]]
这种方法更加类似于mode-covering，倾向于覆盖target distribution的全部模式
第二种方法是使用Reverse KL Divergence
![[Pasted image 20260223161851.png]]
这里的Loss是MaxQ+BC Style Loss
第一种方法不会过度高估，能够进行scaling，但是保持mode-covering，MaxQ+BC的gradient estimation不准确，而且扩展性有限
### Policy Extraction via In-sample Policy Gradient (InPG)
将MaxQ的梯度投影到Weighted BC Gradient
![[Pasted image 20260223163122.png]]
注意这里的数学表达式，让策略朝着MaxQ的方向进行更新
InPG将$\omega(s,a)$ 作为一个神经网络，寻找最优权重，让加权后的BC尽量贴近提升value的方向，这样能在in-sample前提下maxQ
这里的G是gradient vector，通过训练让BC的梯度向量尽量靠近maxQ的向量
==我的理解是，就是通过权重让BC的更新方向与PG的更新方向相同==，这样其实和TrustRegion+MaxQ的思路非常类似，但是能保证In-Sample
$G_{PG}$是沿着MaxQ上升的方向，$G_{BC}$中事先预留好了$\omega(s,a)$也就是我们的训练参数，weighted behaviour cloning，同时要确保这个权重是正数
训练时用的是==L2 Regularization Term==
![[Pasted image 20260224123136.png]]
第一种方法使用的是IVR获得的Weights
![[Pasted image 20260224123605.png]]
也就是这里定义的Weights，发现这种方式会导致Mode-Covering特性，无法达到Optimality
#### Scalable to advanced generative models
可以使用Flow-Matching Models
# Enhancing Online RL with Meta-Learned Objective from Offline Data (AAAI 25)
offline restriction是non-expert policy的时候反而会对性能有负面的影响
构建GILD(Generalized Imitation Learning from Demonstration)，从offline data中提取只是，将motivation提取给optimal policy

Striking a balance between RL and IL实际上是非常困难的，尤其是在offline阶段非常sub-optimal的时候
![[Pasted image 20260224142234.png]]
试图构思一种方法，能够利用sub-optimal的demonstrations，同时也不被限制
### Methodology
方法由两层构成
- Upper level是meta-optimization of GILD
- Lower level是meta-training of RL
#### Overview
GILD控制的就是参数$\omega$ 
GILD创建了一个meta-learned objective $L_\omega^{GILD}(\phi)$ ，从sub-optimal offline demonstrations中蒸馏知识
GILD是依据meta-loss进行更新的
![[Pasted image 20260224150516.png]]
所以说，$\omega$是一个Upper Level GILD Training过程中的可训练参数，这个$\omega$定义了这个Objective Function(Learnable NN)的形状，这个Objective会被作为训练过程中的Loss
![[Pasted image 20260224155441.png]]
可以根据上面这张图来理解，也就是我们的$\omega$的训练目标是，在当前的Loss下，能够训练出尽量好的Critic和Actor模型
![[Pasted image 20260224151716.png]]
MSBE就是Mean-Square Bellman Error，就是最经典的Critic MSE Loss
meta-loss来更新GILD的参数，GILD参数更新的目的是让RL+GILD学习到的Policy优于RL+IL
Lower Level中的Meta-Training就是传统的critic与actor训练，Loss来自于GILD，保证优于RL+IL的情况
### General Imitation Learning Objective
non-expert数据训练出来的policy就会被restricted to be sub-optimal
所以使用General Imitation Learning Objective来进行训练
普通的IL情况下，这里的f往往是一个MSE Loss$Loss^{IL}(\phi) = f(\phi;D^{dem})$ 
GILD则是**使用一个neural network**，学习一个$f_\omega(\cdot)$，生成一个新的Loss:$Loss_\omega^{GILD}(\phi) = f_\omega(\phi, D^{dem})$
GILD采用3-layer MLP
#### Building connection between lower-level and upper-level
Upper-to-lower:
- 在更新RL policy的时候，GILD Loss必须要对policy para $\phi$可微分，也就是input必须要依赖于actor。
- GILD接受**一系列**demonstration pairs和actor action，输出Generalized的IL Penalty
- 
Lower-to-upper
- 为了更新GILD Para
- $Q_\theta$必须要对GILD Para $\omega$是可微分的
### Bi-Level Optimization
Lower-level: meta-training:
- 先收集D，这一部分是通过interacting得到的
- Critic就使用正常的Batch MSBE进行更新
![[Pasted image 20260224162452.png]]
- 在更新actor之前，先pseudo-update actor with ***RL+IL**
![[Pasted image 20260224162603.png]]
- 这里的表达式就是最普通的RL+IL，但是注意并没有对真的参数进行更新
![[Pasted image 20260224162853.png]]
- 接下来依据GILD进行真正的更新，注意这里的GILD中需要来自demonstration集合的一系列数据
Upper-level: meta-optimization
- $\omega$的meta-loss的目的就是让RL+GILD的结果优于RL+IL
![[Pasted image 20260224163255.png]]
这里的Meta Loss是一个==最大化的目标==
而且减去这个IL的Q可以促进学习相对于IL更优的这一部分知识
states是从past experiences中采样得到的
在这里减去IL的分数，可以让tanh的输入处于一个梯度比较大的区域，同时tanh起到了对于梯度的一定限制作用
### Experiments
![[Pasted image 20260225115853.png]]
可以发现TD3+GILD的效果会明显更好
![[Pasted image 20260225120103.png]]
这里可以发现GILD的优化效率更高，且GILD本身的训练速度非常快

# Failure-Aware RL: Reliable Offline-to-Online RL with Self-Recovery for Real-World Manipulation
FARL，针对的是real-world RL
online环境下的实验容易会导致出现irreversible damage，例如breaking fragile objects，或者将物品移出范围等等
FARL尝试最小化IR Failures
提出FailureBench
引入一个safety critic，基于一个latent world model，这些模型都是在offline阶段就可以训练完成的
### Related Work
Safe RL是一个关注RL的安全性与不犯错特性的field，但是Safe RL往往会导致constraints被过早enforced，限制了exploration的性能
Recovery的实际意义应该是post-failure adaptation，而不是preventing failures
FARL的重点是post-training of pre-trained policies in the real world
o2o RL中往往使用conservatism来解决OOD问题
![[Pasted image 20260225135954.png]]
CMDP中的$C(s,a)$是一个只能输出0和1的函数，判断这个state-action pair是否violate了规则
![[Pasted image 20260225141150.png]]
以及这里也会有相对应的Discounted Probability of Constraint Violation
所以CMDP下的RL目标，就是：
![[Pasted image 20260225141609.png]]
### Method
预训练一个Task Policy, 一个Recovery Policy以及一个World Model
task policy使用task demonstrations训练，关注完成task
recovery policy使用recovery demonstrations训练，关注avoid或者escape from near-failure states
world model则会使用task dem与fail dem
Fine-Tuning阶段，微调task policy，由recov policy指导回到$C_H^\pi \leq \epsilon_{safe}$ 
==Online阶段recov policy和world model保持不变==
#### Offline Pre-Training
policy的训练遵循Uni-O4的Pipeline
![[Pasted image 20260225143047.png]]
使用GAE(Generalized Advantage Estimation)，在低方差和低偏差之间找到平衡
Recovery Policy通过BC进行训练，同样采用Uni-O4中的先BC再fine-tune的这个style
**但是在online阶段**，不再调整recovery policy，因为数据量有限
同时训练一个world model，使用task与failure dem同时训练
![[Pasted image 20260225143851.png]]
同时训练reward, value constraint与decoder
这里的world model貌似还同时引入了对latent space的拟合，可以后续再关注一下对world model的学习
==注意几个辅助Network的定义==
*所以这里有三个辅助网络，一个是h，用来将当前时刻的state映射到当前时刻的repr vector, 一个是d，将当前时刻的repr vector和action映射到下一个state的repr vector, 还有一个是decoder S，用来将当前时刻的repr vector映射到这个时刻的state*
#### Online Fine-Tuning
task policy是使用Online PPO进行微调的
使用$\pi_{task}$与环境进行交互，使用world model来检查near-future steps是否安全
![[Pasted image 20260225144823.png]]
如果task policy采样出的action不能保证，我们就会修改task transition为recovery action，也就是会直接采用$a_{rec}$ 
注意每一步的Action都是没有实际采用的，是使用World Model先模拟采取Action之后发生的事情

注意序列式预测的重要性
# 注意这里的审稿意见(OPT)
![[Pasted image 20260208113321.png]]
给出的rebuttal
![[Pasted image 20260208113359.png]]
认为不同的算法本身就对不同的环境有不同的适应性，指出了这个问题

# LOGO
ICLR22
首先和传统TRPO一样生成一个candidate policy，第二步，在一个Trust Region内找到与原来的Behaviour Policy最为接近的Action。
但是level of alignment with the behaviour policy可以逐渐缩小
#### Algorithm
![[Pasted image 20260307155152.png]]
整体的思路非常简单，刚开始的Policy Improvement部分，就是最基础的更新，但是请注意这里的更新同样是考虑了trust region的版本
但是这样的方案存在在reward sparse的时候难以升级的问题。所以现在做出的改进就是，就是在KL-Sphere内找到与原来的policy最为贴合的action
刚开始先设计一个比较大的trust region $\delta_k$ (指的是Policy Guidance这一步的Trust Region)，后面对这个trust region进行decay即可
#### Performance Guarantee
可以求出performance improvement的lower bound
在policy guidance step，初始阶段显然由于offline的guidance，性能会有所提升
### Practical Algorithm
- KL散度是难以直接进行计算的
- 首先定义一个policy-dependent reward function $C_\pi$，定义为$C_\pi(s,a) = \log (\pi(s,a) / \pi_b(s,a))$ 
- 可以证明，策略$\pi$在这个奖励函数下的期望回报$J_{C_\pi} (\pi)$实际上就是平均KL-散度的缩放版本
- $J_{C_\pi} = (1-\gamma)^-1 D^\pi_{KL} (\pi, \pi_b)$ 
- 所以可以推导出(A就是对应的优势值)![[Pasted image 20260308153833.png]]
- 将对$d^\pi$的期望替换为对已知策略$\pi_{k+1/2}$的期望。
- ![[Pasted image 20260308154133.png]]
- 最终可以变为这个表达式，这个表达式实际上就是对KL散度的上界约束
### Experiments
![[Pasted image 20260308155805.png]]
这里可以展示呼整体的学习趋势，LOGO可以在一定程度上参考原来的Expert Knowledge
![[Pasted image 20260308155843.png]]
# Optimistic Critic Reconstruction and Constrained Fine-Tuning for General Offline-to-online RL (OCR-CFT)
NeurIPS 24
传统offline to online RL问题中存在offline到online阶段的Q值Mismatch的问题
提出的思路是在optimistic的方法下重新评估offline policy，从而可以得到optimistic Q-value estimates，防止突然的Q值变化导致的Policy Collapse
使用三种主要的Online RL算法
 - SAC：重点引入Max Entropy Learning
 - TD3：clipped double Q-learning，采用的Action是Clipped Noise
![[Pasted image 20260310133855.png]]
- PPO：最大化的是Action和Reward，Advantage是通过GAE进行计算. $r(\theta)$是两个策略的比值，采用不同的采样方式
![[Pasted image 20260310134008.png]]
### Evaluation and Improvement Mismatches
例如CQL算法在offline阶段pessimistic， 在online阶段optimistic，这样存在Q-value的sharp increase
Improvement Mismatch在Policy-Constraint Methods中是非常常见的，例如TD3BC, AWAC，所以这些情况中更高的Q值不一定能转化为更高的Action Probability.
IQL算法同样存在问题，使用Behavior Policy进行计算
![[Pasted image 20260310142507.png]]
这里就是Offline RL的Unified Definition，就是一个Behavior-Regularized MDP
因为Policy Update经常会依赖于Data Distribution
![[Pasted image 20260310143619.png]]
对比了这里的两个Optimization Objective的区别，体现了offline和online阶段之间存在的evaluation and improvement mismatch
### Method
进行一个policy re-evaluation
- 可以乐观地估计Q值，但是仍然存在unavoidable factors，仍然导致critic和offline policy之间的misalignment
- 提出的方法是align the critic's estimates with the policy's action probabilities
总起介绍三个主要的方法：
- Policy Re-evaluation: 在线微调之前重新评估，得到一个乐观的Critic
- Value Alignment：得到的乐观Critic容易与Actor之间发生Misalignment，所以再次训练Critic让Critic关于Actor进行对齐
- Constrained FT: 解决分布偏移问题
#### Policy Re-eval
offline dataset上训练出来的critic往往会对Q值有比较悲观的估计
如果online阶段直接沿用Critic，那么可能会导致Q-Values的Dramatic Jump
提出OPE方法，off-policy evaluation
使用Fitted Q-evaluation，就是保持Actor，使用离线数据集D对这个策略进行多轮Bellman更新，且不使用Conservative Penalty
一个well-trained policy对于原始的behavior policy足够接近，所以不会有很明显的Extrapolation Error
实验证明，可以对critic进行正确的训练
#### Value Alignment
虽然critic后面已经拥有了optimistic property
但是并没有实现对于offline policy的alignment
misalignment: 最高Q值的action并没有对齐最高的概率
![[Pasted image 20260310192532.png]]
这里可以看出只有Aligned Critic才能正确保持性能
可以看出，re-evaluated critic和offline policy之间的misalignment会导致policy的optimization出现问题。
因为我们默认offlline policy是可靠的，所以想要让critic与offline policy之间保持对齐。
将offline policy actions的Q值当做Anchor，保持不变，而压缩其他的Q值。
##### O2SAC
Intuition: 高概率的动作，Q值估计会更加准确
![[Pasted image 20260310193934.png]]
这里是原来的Q, V和Policy之间关系的表达式, Q'就是我们使用Actor对Critic进行对齐之后的结果
![[Pasted image 20260310194059.png]]
使用这里的表达式对其他transition对应的Q值进行更新
![[Pasted image 20260310194204.png]]
第一个是Align Loss,要求Critic预测的尽量接近我们计算出来的值
第二个是Retain Loss，用于维持系统的乐观性，对齐的是采样出来的Action的Q值，并防止价值坍塌，注意这里的$\bar{\mu}$指的是前面的Re-evaluation之后得到的Critic参数
==注意这里指的都是Value Alignment==部分的内容
##### O2TD3
同样以offline policy的Q值作为锚点，压低远离锚点的Action Q
![[Pasted image 20260310201302.png]]
这里是定义出来的Loss，第一个决定的是align，第二个决定的是最佳动作的对齐
![[Pasted image 20260310201331.png]]
#### Constrained Fine-Tuning
在online learning的过程中，遇到out-of-distribution的现象是不可避免的，可能会导致较大的性能波动。
因为保持了Optimistic的特性，所以OOD state-action pairs上还是会出现Q值的高估问题
又添加了一个constraint term，基于一个divergence的函数f
![[Pasted image 20260310204155.png]]
后面fine-tuning阶段的loss如下所示：
![[Pasted image 20260310204407.png]]
policy就是正常的maxQ+Penalty
Critic就是正常的Bellman Loss
$\lambda$的作用就是限制Penalty的力度大小，有点类似于SAC中的Entropy自动调控项
online阶段的参考policy就是offline阶段训练出来的policy
### Experiment
![[Pasted image 20260310205001.png]]
上面是antmaze环境中的分数
![[Pasted image 20260310205038.png]]
这里可以看到还是不错的，在一些环境中有显著的提升。但是在一些已经saturated的环境上好像也并不是很明显

# Uni-O4
iclr24
核心在于在offline和online阶段均使用on-policy objective，尽量消除offline和online learning之间的gap
offline RL往往会引入Policy Constraints来限制对于OOD data的访问
从offline到Online的转换就会涉及到这里的instability
这些方法都会导致Initial Performance Drop, 或者是poor asymptotic performance
提出了==fine-tune stability与asymptotic performance==之间的tradeoff
提出的方法是on-policy optimization method
使用Offline Policy Evaluation stage来评估updated policy，进行multi-step policy improvement
#### Offline Policy Evaluation
有很多种offline阶段的Evaluation方法，需要将offline dataset划分为training set和validation set
![[Pasted image 20260311192409.png]]
训练一个estimated dynamics model
训练的是Maximum Likelihood Objective
Approximate Model就是训练出一个Dynamics Model，可以进行Monte-Carlo Rollouts
Fitted Q-Evaluation是一种value-based的方法，拟合Bellman Operator来学习Value函数，在offline数据上最小化MSBE
### Method
整个过程分为三个阶段
- Supervised Learning
- Multi-step policy improvement
- Online fine-tuning
![[Pasted image 20260311210308.png]]
这里就是整个Pipeline
#### Ensemble BC with disagreement-based regularization
在off-policy evaluation中尽量避免extra conservatism
首先recovering收集offline dataset的behav policy $\pi_\beta$ 
直接使用Behaviour Cloning来重新构建，但是**容易会导致mismatch**，因为dataset往往是使用多个policy来进行采样的。
所以提出了**Ensemble Behavior Cloning with disagreement regularization**
要学习的是a set of policies，且要增进他们之间的diversity
![[Pasted image 20260311195401.png]]
上面的Proposition 1给出了两个分布之间的距离度量，这里关注的两个概率分布是某一个具体Policy和Ensemble Policy
以及下面给出了KL Divergence的下界，这个下界是具体可计算的
![[Pasted image 20260311195606.png]]最终的优化目标就是：
![[Pasted image 20260311200606.png]]
前面的部分是Likelihood，后面的部分是不同Policy之间的Divergence
以及注意Combined Policy的具体定义式，在上面Prop.1的最后一行
这里的正则化项$Z(s) = \int_a da \,{max}_j \pi_\beta^j(a|s)$ 
这一阶段的Q和V使用的是IQL的Style进行计算
### Multi-Step Policy Ensemble Optimization
提供了一个简单的offline policy evaluation方法
![[Pasted image 20260311202616.png]]
这里使用了PPO中相同的clipped surrogate objective
这个对于每个ensemble policy中的成员是**独立进行更新的**
这里的采样分布来自于**离线数据集**
下标k代表的是策略迭代的论述，参考的旧策略，只有在新策略通过AM-Q考核的时候，基准策略才会进行更新
原本的Advantage值是**使用GAE进行估计的**，现在**改为使用AM-Q**
这里的更新style整体上与PPO是非常类似的
所以这里**使用新的policy进行采样**而不是k=0就不会进行过度的clip
##### Offline policy evaluation for multi-step policy improvement
OPE方法结合了Approximate Model与Fitted Evaluation
![[Pasted image 20260311204915.png]]
AM-Q就是将地平线长度内的Q值叠加起来
![[Pasted image 20260311205158.png]]
这里就是具体讲上面假设的optimal和true model转化为我们的估计，这个估计的计算方法在上面提到过
![[Pasted image 20260311205405.png]]
我们能够证明出这两个J之间的差距是有上界的。

只有在我们确定现在迭代后的Policy的J值大于原来的J值的时候，才对Policy进行实际的更新。
同时这里的Advantage值同样使用AM-Q方法的Q-V来进行计算
最终one policy is chosen by quering OPE
使用AM-Q来进行评估，也可以选择top-k policies
##### Online PPO fine-tuning
offline阶段训练出来的value function和policy直接作为online PPO的initialization
### Experiments
![[Pasted image 20260311210119.png]]
这里看的话感觉结果还可以吧，有一些的分数好像并不是特别高