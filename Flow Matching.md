# Flow Q-Learning
是一种offline RL方法，将flow-matching应用到Policy的设计上
offline RL的本质：在dataset的state-action distribution的限制下maximize returns
并没有直接训练flow或者diffusion policies来最大化value的方法
**Behaviour-Regularized Actor-Critic**
- 最简单的offline RL frameworks，引入了Behaviour Cloning的Loss
![[Pasted image 20260210151008.png]]
- 也就是这里在policy中添加了这个penalty
![[Pasted image 20260210151600.png]]
这里是对Training Objective的改进，直接让velocity变成这个差距，相当于是直接依赖于linear path
FQL这里相当于就是将原论文中OT的速度转化为了$\sigma_{min} = 0$的情况
$\mu_\theta就是\psi_\theta(1,s,z)$ ，直接将noise映射到Action(通过ODE过程)，注意z就是noise $z_0$ 
注意这个Policy本身是deterministic的，但是初始噪声是具有随机性的，所以Policy也可以认为是具有随机性的
### Flow Q-Learning
Desiderata are twofold: 首先是需要flow-matching的expressiveness，且希望使用也更加简单
最简单的方法也许是将BC Loss换成Flow Matching Loss
![[Pasted image 20260210154530.png]]
这里前面的是critic loss，而后面的就是FM Loss，实际上就是Behaviour Cloning的Loss
但是flow objective需要numerical ODE Solver的反向传播，容易导致不稳定性
计算Q的梯度的时候，要求action对theta的梯度
#### Solution
直接使用BC Loss，随后再训练一个分离的，one-step policy来最大化value function，同时使用关于BC flow policy的distillation loss
这个新的policy就是$\mu_\omega(s,z)$，同时从pure noise开始生成action
![[Pasted image 20260210160705.png]]
这里就是Distillation Loss，使用的是MSE/L2 Loss
有三个主要部件，critic, BC Flow Policy, One-Step Policy，分别$Q_\phi, \mu_\theta, \mu_\omega$
首先使用BC训练FlowPolicy，接下来就使用传统方式训练critic
这里的这个Loss是始终存在的
# Flow Matching with Injected Noise for o2o RL
iclr 26
经常会将Diffusion或者Flow Matching作为一个从state到action的Generator
FINO将噪声注射进flow matching，保证了在offline pre-training阶段的diversity，确保学习更加广泛的action space
==OGBench&D4RL==
Flow Q-Learning中，policy design中设计导了flow-matching，使用Behavior Cloning作为Flow-Matching的Loss，但是这里使用的是one-step policy

Flow Matching训练时用到的数据就是通过uniformly-distributed的时间变量进行插值，vector field的训练目的就是拟合这一个linear path
使用Euler Method来进行ODE Solving

# Flow Matching For Generative Modeling
- Generative Modeling
- 建立在Continuous Normalizing Flows上
- 之前的Diffusion Model更多建立在Gaussian Probability Paths上
CNF(Continuous Normalizing Flow)的目的就是为了建模**随机概率路径**
- Flow Matching Objective的目的就是回归到一个target vector fields
### Continuous Normalizing Flows
两个重要的objects:
- Probability Density Path：一个time-dependent probability density function
- 且应该满足$\int p_t(x) dx = 1$
- 还有一个time-dependent vector field
- 这个vector field $v_t$可以用来构建一个Diffeomorphic Mapping(微分同胚映射)，意思是存在一一对应，连续性与逆映射的连续性。同胚性保证了不破坏物体的原结构，又能保证变形结构的流畅性
- 一个flow的全过程可以使用基于Vector Field的ODE，且t = 0时的$\phi_0(x) = x$ 
- flow的定义就是这个微分同胚映射
可以总结一下：
- Probability Density Path: 整个群体在某一个时间分布的概率地图，可以认为这个概率分布中的一个点就是一个特定维度的向量，p(x)输出的就是这个点的概率
- Vector Field: 描述空间中每一个点(也就是一个向量)在某一个瞬间的运动趋势
$$
\frac{d}{dt} \phi_t(x) = v_t(\phi_t(x))，其中\phi_0(x) = x，该表达式的右侧乘上dt表达变化
$$
- 上面这个是Flow($\phi_t(x)$)的定义式，flow的微分就是当前点的vector field，也就是$\phi_t(x)$就是某一个数据点x在经过时间t之后最终到达的位置
- Flow就是一个Diffeomorphic Mapping
- 可以使用神经网络来建模vector field $v_t$，构建的是Continuous Normalizing Flow
- CNF可以用来将一个简单的pure noise分布转化为一个complicated $p_1$
- 首先定义一个push-forward 算子，定义为
$$
[\phi_t]*p_0(x) = p_0(\phi_t^{-1}(x)) det [\frac{\partial \phi_t^{-1}}{\partial x}(x)]
$$
- 接下来可以定义$p_t = [\phi_t] * p_0$ ，这里用到了行列式的基本知识，利用原来的概率密度乘上一个scaling ratio
### Flow-Matching
- $x_1$是一个random variable，我们假设我们能从$x_1$对应的概率分布进行采样，但并不能直接了解到density function，同时我们由一条$p_t$作为概率分布函数
- 假设$p_0$是一个正态分布，$p_1$是一个给定的q函数
- Flow Matching的目标就是构建这个flow的过程
- 如果已经给定了一个target probability density path，以及一个vector field，那么loss就是
![[Pasted image 20260209155819.png]]
- 其中的参数$\theta$来自于vector field，$u_t$是一个vector field，u是真实理想的速度场
- vector field就是速度
#### Constructing $p_t, u_t$ 
- 使用$p_t(x|x_1)$来代表在$x_1$条件概率下的x分布
- 让t = 0的时候的条件概率$p_0(x|x_1)$就设置成p(x)，t = 1的时候是一个聚集在中间的分布，例如正态
- $x_1$对应的是t = 1时刻的样本
- 那么有$p_t(x) = \int p_t(x|x_1) q(x_1) dx_1$
- t = 1的时候$p_1(x) \approx q(x)$
- 可以创建一个marginal vector field
$$
u_t(x) = \int u_t(x|x_1) \frac{p_t(x|x_1)q(x_1)}{p_t(x)}dx_1
$$
- 创建了Conditional VF和Marginal VF之间的关系
- 如果知道了$u_t(x)$，那么就可以构造出$p_t(x|x_1)$ 
- 注意这里构建的都是Marginal Distribution，通过条件路径来构建复杂的边缘路径
- 某个时刻的向量场通过所有条件向量场进行加权平均
#### Conditional Flow Matching
首先定义Conditional Flow Matching Objective
![[Pasted image 20260209163609.png]]
$x_1$仍然是从$q(x_1)$中采样得到的，而且可以证明这个Loss和FM情况是等效的，也就是$\nabla_\theta L_{FM}(\theta) = \nabla_\theta L_{CFM}(\theta)$ 
## Conditional Probability Paths and Vector Fields
- 讨论如何构建$p_t(x|x_1)$以及$u_t(x|x_1)$
- ![[Pasted image 20260209165801.png]]
- 以这样的形式构建Probability Paths
- 假设$\mu_0 = 0, \sigma_0 = 1$，也就是一个纯粹的Gaussian Noise
- $\mu_1(x_1) = x_1, \sigma_1(x_1) = \sigma_{min}$
- 有无数种构建这个probability path的方法，
$$
\psi_t(x) = \sigma_t(x_1)x + \mu_t (x_1)
$$
- 在x是标准高斯分布时，这个仿射变换能将分布转化为一个特定的高斯分布，所以这个表达式能让noise distribution变成$p_t(x|x_1)$，psi的意义就是表达在时间t的数据点全体分布
- 所以$[\psi_t]_* p(x) = p_t(x|x_1)$
- x是噪声分布，且有
$$
\frac{d}{dt}\psi_t(x) = u_t(\psi_t(x) | x_1)
$$
- 这个表达式的作用是生成对应的$u_t$，也就是向量场
最终可以得到最后的CFM Loss
![[Pasted image 20260209171438.png]]
Reparameterizing：可以将一个概率分布建模为：（原始概率分布+其他参数，例如调整标准正态分布的均值和方差）
![[Pasted image 20260210122146.png]]
这里是$u_t(x|x_1)$的推导过程，$u_t$就能生成对应的Gaussian Path
### Special Instances of Gaussian Conditional Probability Paths
#### Diffusion Conditional VFs
Diffusion Models从数据点开始逐步增加噪声知道变成pure noise，这是一个stochastic process
可以注意到Score Matching的效率不够高，仍然不会走最优的路径
#### Optimal Transport conditional VF
![[Pasted image 20260210124835.png]]
直接使用最为简单的传输路径，可以推导出当前case下的CFM Loss
![[Pasted image 20260210124907.png]]
可以看出OT方法的效率明显更高，是直接进行转换的
![[Pasted image 20260210125053.png]]

# Flow Matching with Injected Noise for o2o RL (iclr 26)
o2o RL的目的就是为了在online阶段能够快速提升性能，所以我们应该重点关注online interaction开始之后能否迅速提升性能
传统的FQL中需要训练一个One-step policy
![[Pasted image 20260223115737.png]]
注意这里的超参数$\alpha$设定有时候是非常大的
### Motivation
offline RL的性能会受局限于原本数据集的规模，试图找到一种方法，在不增加数据集规模时也能提升性能
在offline pre-training phase引入injected noise，在flow-matching过程中加入噪声，帮助flow model在一定程度上增强扩展性，接下来再从这个Perturbed behaviour model中进行蒸馏，某种意义上可以促进broader coverage
### Method
接下来介绍FINO
- 在offline阶段，加入controlled noise，促进policy的exploring能力
- 在online阶段，使用这个expanded action space
#### Noise Injection for Flow Matching
![[Pasted image 20260223122357.png]]
FQL中的$\sigma_{min}$被设置为0，这样会导致distribution坍塌到individual data points，导致coverage非常狭窄
![[Pasted image 20260223123007.png]]
现在提出一个新的Loss Function，要拟合的目标变成了一个有噪声的项，同时前面的Path也加上了一个噪声，且注意这里的noise是一个scheduled variance
这里的Noise Scheduling可以确保噪声逐渐增加，确保动作空间具备足够的厚度
前面的是vector field，vector field的输入中添加了噪声
拟合的目标是$x_1 - (1 - \eta)x_0$，确保最终生成的目标方差为$\eta^2$
![[Pasted image 20260223124028.png]]
同时，使用数学推导证明了
- mean不变
- 方差会比原始flow matching更大
- 该推导与传统Flow Matching是等效的
![[Pasted image 20260223124717.png]]
这里就是pseudocode的部分，其实整体与FQL是非常类似的
这里是一个dataset上生成action的例子
![[Pasted image 20260223125145.png]]
可以发现FINO方法生成的覆盖范围明显更加广阔
### Entropy-Guided Sampling
首先可以从多个base noises(也就是$x_0$)中进行采样，取多个candidate actions，接下来使用类-softmax进行抽样
![[Pasted image 20260223130134.png]]
从这个分布中采样action来确保一定的exploration
同时注意这里的$\xi$，越大说明更加greedy，越小说明越explore
所以这里同样有一种Entropy Tuning方法，维持我们的Entropy在Target附近
![[Pasted image 20260223131019.png]]
$\alpha_\xi$是负数，据这个公式可以进行调整，如果当前熵大，那么$\xi$增加降低熵
![[Pasted image 20260223130930.png]]
这里说明用的也是OGBench
### Practical Implementation
Entropy并不恩能够直接计算，所以这里使用Gaussian Mixture Model来计算Entropy，可以用来估算Entropy
$\eta$决定了Injected Noise的方差，设定为0.1
将number of sampled actions设定为action dimension的一半
### Discussion
#### Noise Injection Point
- 对比与直接对Action加上Noise的方案，可以发现FINO的性能会更加理想
- 而且在door-cloned上可以发现直接加上噪声反而会导致性能下降。
![[Pasted image 20260223142443.png]]
后续能证明简单的Noise-Scaling的效果比不了FINO
同时后面也做了实验，分别w/o Noise或者w/o Guidance
![[Pasted image 20260223143410.png]]
