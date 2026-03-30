Model-Based Offline-to-Online Reinforcement Learning
Policy能够进行Fast Adaption to online environment是非常重要的
我们的方法可以动态调整offline与online data，为的就是丝滑的online adaption
# 1. Introduction
现在offline-to-online transfer过程中始终存在问题
问题主要来自于ineffective control of distribution shift in existing methods
Distribution Shift存在的原因：
- offline data的来源是若干个policy给出的运动轨迹
- online training stage中的transitions都是由currently learned policy采样而来的
更好的性能需要**同时专注于online performance与offline performance**
我们因此提出了MOORe这种算法
- 改进了Data Sampling Scheme
- 鼓励了smooth transfer in the early online stage
- 鼓励了fast adaption in the late online stage
主要Contributions
- 使用model-based来解决o2o RL
- 使用Prioritized Sampling Scheme
# 2. Related Works
Offline RL
- 存在Extrapolation Error与Distribution Shift两大主要问题
- 解决方案主要是Q-Constraints
- 也有使用Model-based methods来学习Conservative Policies的方法
	- 就是在模型生成的Transition上施加惩罚来保证更好的保守性质
o2o RL
- 也引入了很多的Constraint方法来限制RL算法的更新幅度
# 3. Method
## 3.1 Preliminaries
在Model-Based RL中，一个Environment Dynamics Model $\hat{M}$，会包含一个Transition Function与Reward Function

有关于MOPO：
![[Pasted image 20250829163157.png]]
这里的带hat的都是由我们自己的Model预测出来的值，u是一个Uncertainty Term，**是通过Model Ensemble技术**进行预测的
$D_{on}$就是存储了online interactions的dataset，$\pi_{on}$就是同时从$D_{on}与D_{off}$中学习到的Policy，$M_{off}$和$M_{on}$就分别是offline online stage中学习到的dynamic model.
- Model Ensemble是训练多个Environment Model，然后根据这些Model之间的差别大小来决定**不确定性**的工具。
## 3.2 Model-Based o2o RL
Distribution Shift是o2o RL的Main Concern
最简单的方法是直接将policy, value function 与 env model转化到online stage，但是这样效果很差。
我们的方法是，让offline与online transition拥有不同的priority
![[Pasted image 20250829164828.png]]
在属于Online数据集的时候拥有**固定优先级**，Offline数据集的时候根据f(d, t)决定，$d = (s, a, s', r)$ 
![[Pasted image 20250829165425.png]]
$\alpha$是一个给定的常数
t就是epoch数，t越大那么Priority这个参数相对应地就会更小。
![[Pasted image 20250829165543.png]]
### Pseudo-Code解释
***Offline Stage***
- 我们先使用MOPO算法进行的初始policy训练，以及初始的Env Model, Q-Function，因为默认使用的是SAC算法，所以我们最终可以得到一个policy与value function.
***Online Stage***
- 直接迁移**Policy, Env Model, Q-Func**
- 初始化$D_{on}, D_{Model} = \Phi$ 
- 开始Epoch循环
	- 先设定$D_{off}$的优先级
	- 小循环
		- 与真实环境交互，得到一个transition
		- 计算这里的Pr(d, t)
		- 如果到了给定的步长
			- 在$D_{off} \cup D_{on}$上进行Prioritized Sampling，训练我们的**环境模型**直到收敛。
			- 挑选一些数据集中的states
			- 使用模型进行Rollout
			- 加入到$D_{Model}$
		- 使用$D_{model}$上采样的数据进行训练
![[Pasted image 20250829194744.png]]
- 请注意，这里是***Prioritized Sampling(PER)*** 的具体实现方法，存在超参数调节偏向程度。
- 所以可以这么理解，**直接的来源**，也就是实际进行梯度更新的transition数据确实都是**来自于Model生成的**，但是这个Model是根据我们前面的Offline dataset与Online dataset的并集**学习**出来的
- 很重要的是，我们的Model是在随着时间演进不断更新的，需要重新学习我们数据集对应的分布。
### 主要特性
#### Smooth Transfer
- 首先，online transitions还很少，且这里的Pr值因为t小还很大，大概率**仍然会使用offline dataset**来进行训练。
- 正因如此，在Online Stage的起始，训练的大部分数据仍然是Offline的，所以Distribution Shift还非常小。
#### Fast Adaption
- 在后期，offline data的权重越来越小，我们会使用pure online data来进行训练
- 同样，Uncertainty Term会大幅度减小，Conservative性质会降低
- 所以到最后我们的算法就会变为传统的MBPO-like Online RL Style
## 3.3 Theoretical Foundations
## 3.4 Influence of Conservatism
- Offline RL需要训练一个Conservative Manner，但是online RL不需要Conservative
- 但是如果在online stage一开始就直接Abandon Conservatism会伤害Transfer Smoothness，如果一直Conservative可能会损坏效率。
- 虽然我们没有改变Penalty Term的表达式，在online training stage中我们的uncertainty term会自己下降，所以我们的算法也就会慢慢变成传统online RL。
# 4. Experiments
![[Pasted image 20250829202352.png]]
## 4.1 Main Results
在绝大多数情况下，MOORe都比其他的o2o RL算法表现会更好，**收敛速度更快**。
其次，我们的Learning Curve也会更加光滑，说明了我们的算法在转换到online training stage的时候具有更平滑的转换。
- 可以总结：我们的MOORe算法可以解决fast adaption and smooth transfer
我们可以观察到在Random Benchmarks上，randomness可能会导致在offline阶段的collapsed policy
## 4.2 Ablation Study
### Prioritized Sampling Schemes
对比
- Uniform Sampling，这里全部的action都是相等概率的
- half-half sampling就是一半来自于online一半来自于offline
- pure online：全部来自于online interaction
这里的half-half与pure online都会遇到在online的初始阶段的Performance Drop
而Uniform Sampling虽然非常smooth，但是难以快速适应online data
Transfer Smoothness->online stage一开始有没有缓缓从offline过度到online(避免性能下降)，也就是我们有没有
Fast Adaption->后面有没有好好利用珍贵的Online Data
## Robustness to Priority Hyper-param
![[Pasted image 20250829205725.png]]
前面有一个决定Priority的超参数$\alpha$
可以看出我们训练效果对这个超参数并不是很敏感
## Effect of Penalty Coefficient
我们之前提到过，MOPO中的conservatism term可能会有副作用。
![[Pasted image 20250829210030.png]]
- 我们的Uncertainty是**越来越下降的**
- 可以看出我们的online policy并没有受到Conservatism Term多大的影响。
- 不同epoch之间的保守性差异是非常小的，说明我们的模型保守性
- 可以看出我们这里的uncertainty value error并不大，所以Transfer Smoothness可以被保证。
# 5. Conclusion
我们的算法可以
- 完成Smooth Transfer：从offline平滑转向online，包括conservatism的平滑转化
- 可以Improve Efficiently：可以在Online Setting中根据Dynamic的Priority Function开始高效率应用Online Data。