# Is Value Learning Really the Main Bottleneck in Offline RL (nips 24)
有多重Policy Extraction的实现方式，Policy Extraction指的就是通过给定的Q函数训练一个Policy来达到Value Maximization的目的
AWR: Advantage Weighted Regression，不直接求Q的梯度，相当于对样本中的Advantage值，进行有权重的加权模仿
在offline RL中有三种方式来进行Behaviour Constraining
### Main Hypothesis
有三个最主要的因素
- value function估计的误差
- value function得到的policy的问题
- 不完美的generalization
这里提出的要点是，policy learning才是影响性能的最大因素，包括extraction与generalization的问题
Behaviour Policy可能是我们自己建模得到的结果
### Value or Policy
- 使用的baseline算法都是分离了value与policy的算法，发现这类方法，Policy Extraction带来的影响都会更大
使用三种算法：IQL, SARSA, Contrastive RL
- IQL拟合一个Expectile Q
- SARSA拟合的是Behavioral Q function，也就是给出的数据集中的Q函数
Policy Extraction的方法
1. Weighted Behavior Cloning: 使用Advantage的指数作为权重进行BC
2. Behaviour-Constrained PG，例如TD3BC，在原本maxQ上，加上BC Loss
3. Sampling-Based: 从behave policy中采样，取这里面的最大值
![[Pasted image 20260222115725.png]]
![[Pasted image 20260222134550.png]]
这里可以发现
经过对比可以得出DDPG+BC是最好的Policy Extraction方法
以及可以发现，Policy Extraction的方法会对data-scaling的趋势造成较大的影响
Policy Extraction在大多数情况下比Value Learning更加重要
#### Why DDPG+BC is better
AWR学习到的Actions就是在dataset actions的convex hull内
而DDPG+BC会对value function进行hillclimb，同时限制不能太远
AWR对于High-Adv的样例过于敏感，容易导致Overfitting
AWR在样本有限时，会增加所有样本的似然情况
### Policy Generalization
![[Pasted image 20260222131818.png]]
这里引入了三个MSE的概念
- Training MSE是在训练数据集上的Loss
- Validation MSE是在验证数据集上的Loss
- Evaluation MSE是在实际访问transitions上的loss
o2o RL中的online阶段，只会降低evaluation MSE
- 所以最重要的往往是policy是否能够在自己的state distribution上进行正确的泛化
可以发现当前的方法已经能够拟合in-distribution states，主要问题来自于如何进行Generalization，控制novel states上的情况
我们能给出多种解决方案：
1. 提升offline data coverage，尽量使用覆盖范围大的数据集
	- 这里可以发现state coverage比数据集的optimality更加重要
2. Test-time policy improvement
	- 如果不能改变offline dataset，那么就只能使用下面这些方法
	- On-the-fly Policy Extraction: 直接在eval的时候，在action上也添加上这个对应的梯度项![[Pasted image 20260222170557.png]]
	- 但是注意并没有改变本来的Policy Network
	- 接下来就是Test-Time Training方法，这里对参数进行了更新![[Pasted image 20260222170709.png]]
### Conclusion
- 可以发现policy实际上是对性能的bottleneck
- 主要是policy extraction与generalization
- 所以应该保证diverse data,以及允许policy利用好value function
- 指出DDPG+BC是最好的Policy Extraction方法

## IQL
- Offline Reinforcement Learning
- 之前的方法往往是in-distribution constraints，IQL尝试进行的是in-sample learning
- offline RL要求最小化的目标：(实际上就是TD Error)![[Pasted image 20260202131822.png]]
- 注意使用的是Target Network，以及这里使用的这个Q值永远考虑的都是Optimal的状态，下一个Q取的都是Optimal Q
### Details
- 在TD Loss中完全方式避免out-of-sample的样例
- 这里是SARSA-Style的Loss![[Pasted image 20260202132040.png]]
- 这个表达式中直接利用轨迹中实际存在的下一步操作来，使用MSE Loss进行梯度下降，而Q-Learning则是结合了对于最优的估计。
- ![[Pasted image 20260202133004.png]]
- 这个是value function的loss
Expectile Regression
- 估计的对象是分位数，而不是一个确定的值
- Quantile和Expectile之间的区别是不同的定义方式
- 引入了Expectile Regression之后，我们的Learning Loss发生了改变![[Pasted image 20260202142151.png]]
- Value函数的作用转化为估计一个较好的值(基于Expectile Regression)
现在分析一下三个网络的Loss与训练目标
Value网络：
- 估计当前state的整体价值(偏向于那些价值更高的动作)
- ![[Pasted image 20260202143610.png]]
- 这里就是依据当前数据集的数据(体现offline)进行训练，V网络对Q值进行Expectile Regression
- $L_2^\tau$ 体现了考虑权重的程度
- $L_2^\tau = |\tau - 1(if \;u < 0)| u^2$
- 所以V就是拟合一个加权之后的Critic
Critic网络：
![[Pasted image 20260202144009.png]]
- Critic同样从offline dataset中采样，拟合的是当前动作的reward与下一个state的value，避免了取max的操作导致外推误差
Policy Extraction:
- Policy网络并不会参与到critic与value网络的训练中
![[Pasted image 20260202144430.png]]
- 这里就是Policy的Loss，这个L是要进行最大化的目标，我们要加大这些更优动作的概率
- beta越小，熵就越大(因为优势的动作优势更小)
## SAC
- Maximum Entropy RL
- 原本的最大化目标：
$$
\sum_t E_{(s_t, a_t \textasciitilde \rho_\pi)}[r(s,a)]
$$
现在的最大化目标变为
$$
\pi^* = argmax_\pi \sum_t E_{(s_t, a_t \textasciitilde \rho_\pi)}[r(s_t,a_t) + \alpha H(\pi(\cdot | s_t))]
$$
Soft Policy Iteration
- 就是考虑了Entropy之后的Policy Iteration，需要定义一个新的Operator
![[Pasted image 20260202154329.png]]
- 而这里V的定义就是考虑了Entropy的
![[Pasted image 20260202154427.png]]

Details of Soft Actor-Critic
- $\theta$是Q-function的参数，$\phi$是policy的参数
首先是soft Q-function要最小化的目标
- ![[Pasted image 20260202155114.png]]
- 注意这里用到了State Value，定义如下：![[Pasted image 20260202155149.png]]
- 注意这里的state value是依据当前状态下的policy来进行采样的，而不是默认就取optimal
接下来是Actor要最小化的目标
![[Pasted image 20260202155505.png]]
- 这个表达式中，要最大化Q+Entropy，依赖的概率分布是对当前Policy而言的概率分布
- 实际上就是最小化KL散度，原始表达式如下![[Pasted image 20260202155846.png]]
- 可以转化为J的表达式，等效于最小化上面这个式子，上面这个式子，就是让当前policy的分布与Critic经过Exponential Normalization之后的分布之间具有最小的KL-Divergence
# Network Sparsity Unlocks the Scaling Potential of DRL
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
