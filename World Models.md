# World Models
目的是建造generative neural network，学习压缩后的环境表示(both spatial and temporal)
可以完全在hallucinated dream内进行训练，再直接应用到actual environment中
Credit Assignment Problem: 强化学习中，如何确定海量参数中哪个参数，或者哪个动作真正对reward造成了贡献，在参数量大的时候尤其明显。
所以分离出来一个大模型学习环境特征，一个small controller用来生成policy，防止credit assignment问题
### Agent Model
- Visual Sensory Component: 压缩视觉输入到一个representative code
- Memory Component: 能够根据historical info预测future codes
- Decision-Making: 基于vision与memory component给出的向量，做出决策
![[Pasted image 20260226115652.png]]
可以总结为V, M, C三个基本组件
#### 1. VAE(V) Model
- 如果玩游戏，那么输入就是一个2D Image Frame
- V Model目的是学习一个abstract, compressed representaion
#### 2. MDN-RNN (M) Model
- 期望**将过去发生的事情进行encode**
- M Model的作用就是要predict the future
- 作为一个predictive model of the future Z vectors(也就是V本来要输出的内容)
- 训练RNN输出一个density function
![[Pasted image 20260226121332.png]]
- 使用Mixture Density Network训练，Combined with RNN
- 输出的概率分布是**GMM高斯混合分布**，也就是多个高斯分布的混合，包括各自的均值与方差，描述了一种多中心的复杂概率分布
- MDN是一种建模复杂概率分布的神经网络，能够处理环境中的离散随机事件，MDN就是用来输出GMM的参数的，避免了 **“平均化错误”**
- 生成的是$z_{t+1}$和$h_{t+1}$，分别是生成的Hallucination和RNN Representation
#### 3. Controller(C) Model
- 就是用来决定Actions的，目的是最大化cumulative reward
- 规模尽量小，且与V和M分开训练，确保Controller不会有Credit Assignment的问题
![[Pasted image 20260226123340.png]]
这里的$z_t和h_t$分别是V输出的latent vector与M输出的Memory Vector，在时间步t的隐藏状态向量
### Putting V, M and C together
![[Pasted image 20260226123646.png]]
首先由V处理raw observation，生成对应的$z_t$，拼接M生成的h，输入给C生成对应的Action，与环境交互，如此循环
V和M可以尽可能复杂，以增强表达能力。
而C应该尽可能简单，甚至可以使用Evolution Strategies来训练
使用CMA-ES(Covariance-Matrix Adaption Evolution Strategy)来训练
数值优化，撒下一群探测器，进行解的选择重组并自适应调整。
CMA-ES只需要提供最终的累计奖励，不关注动作序列的完整历史
### Car Racing Experiment
首先使用dataset，VAE-Style训练一个Vision Decoder，使其能够正常生成z
接下来使用$a_t, z_t, h_t, z_{t+1}$训练一个$P(z_{t+1} | a_t, z_t, h_t)$ ，也就是M Model
End to end就是将全部组件一起训练，使用一个Loss将这个Loss回传到每一个部分
world model指的一般是V和M，V和M对现实的Reward Signal完全不知道，仅做压缩与预测的工作
#### Procedure
1. 从random policy中收集10000 rollouts
2. 训练VAE
3. 训练MDN-RNN来训练$P(z_{t+1} | a_t, z_t, h_t)$
4. 定义Controller
5. 使用CMA-ES训练Controller，最大化cumulative reward
#### Experiment Results
**V Model Only**
- 如果只使用V，训练效果不佳
**Full World Model**
- 将$z_t, h_t$均作为传递给Agent的参数
- 性能会更好
- 暂时不需要plan ahead，只是利用了$h_t$的表征能力，就出现了明显的性能提升
#### Car Racing Dreams
可以使用M生成$z_{t+1}$的概率分布，采样并假设这个就是下一个时间的real observation
让C直接在hallucinated environment内部交互即可

### VizDoom Experiment
试图从hallucination中学习，再应用到actual environment中
使用VizDoom环境作为实验环境，避免被火球打中
M的训练只用于训练next $z_t$
同时添加一个量代表是否下一帧死掉了$d_t$
simulation中不需要V，直接在latent space中训练即可

#### Training Procedure
- 使用Random Policy采10000rollouts
- VAE训练，使用V将图片转化为latent space
- 使用MDN-RNN来建模$P(z_{t+1},d_{t+1} | a_t,z_t,h_t)$
- 训练Controller，在virtual environment内部进行训练

可以发现Controller可以在dream environment内学习到正确的策略
- 连不能穿越墙面，决定是否被杀死等因素，都是神经网络自己学习到的特征
- 调节生成$z_{t+1}$过程中的温度参数$\tau$会影响新生成环境的效果，甚至可能就会直接死掉(说明仍有不足)
- 虽然V Model不能掌握全部details，但是agent仍然可以从中学习
- 一个更多噪声的环境可以让Actor在Original, Cleaner Environment中有更好的学习效果
#### Cheating the World Model
出现了Adversarial Policy，可以用特定的方式移动，防止火球生成
因为world model仅仅是approximation，所以偶尔会生成不符合actual laws的轨迹
所以world model会对controller而言变得exploitable

现在让Controller同时可以访问M的Hidden States，得以访问internal states而不是observations，这样agent就可以操纵hidden states，找到Adversarial Policy

我们的解决方案：试图让C Model更难exploit the deficiencies of the M model。使用MDN-RNN作为Dynamics Model，使得有一定的随机性，随机性也可以通过调节温度超参数$\tau$来调节，$\tau$更大的时候随机性更强，具有更强的Realism，减轻Exploitability
如果使用较低的temperature，那么就类似于deterministic LSTM，容易出现Exploitation
![[Pasted image 20260226153055.png]]
虽然温度更高的时候更难出现adversarial policies，但是如果温度过高可能导致完全无法学习。
### Iterative Traning Procedure
World Model可能是很难学习的，且只有一部分是可访问的，所以World Model也要在Iterative Training中不断加强性能
1. Initialize M, C
2. Rollout，将observations与actions进行存储
3. 训练M建模$P(x_{t+1}, r_{t+1}, a_{t+1}, d_{t+1} | x_t, a_t, h_t)$，训练C在M内试图优化
4. 返回2 if task not completed
多轮循环可以在复杂任务中提升world model性能
可以使用M的Loss Function来鼓励Real World Agent访问不熟悉的区域(M的Loss大的区域)，从而提升World Model性能
这里的M Model还添加了对action和reward的预测
可以使用M模型，吸收如何平衡，如何迈步这些低级运动技能。
我们让world model来向C学习，world model中已经吸收了，可以认为这些信息都蕴含在向量$h_t$中
![[Pasted image 20260226160939.png]]
==这一段还是不太理解？？==
是不是可以理解为M借助预测C，本身也学习到了环境的一些本质特征，M学习了物理环境的特征，而C也不需要再反复学习低级的运动特征

论文阅读进度
World Models
Flow Matching with Injected Noise for o2o RL
Uni-RL (nips 25)
Enhancing Online RL with Meta-Learned Objective from Offline Data (AAAI 25)
Failure-Aware RL: Reliable Offline-to-Online RL with Self-Recovery for Real-World Manipulation
Is Value Learning Really the Main Bottleneck in Offline RL (nips 24)

# Dream To Control: Learning Behaviours by Latent Imagination
dreamer v1, iclr2020
### Intro
- 使用latent states可以在极大程度上降低对运算资源的要求
- 通过past experiences构建对world model的representations
- 一般使用imagined rewards最大化的方法
- Dreamer V1的特点是可以学习long-horizon behaviours，以及引入了actor-critic算法，而不是依赖于derivative-free optimization methods
- 在latent imagination内学到action与value
Learning Long-horizon behaviour by latent imagination
Empiricial Performance for visual control
### Control with World Models
Agent需要在latent space中预测hypothetical trajectories
使用latent trajectories学习action and value models
指出了最为核心的三个环节：
- 学习动力学模型
- 在想象中进行行为学习
- 与环境交互
提出Latent Dynamics的结构
![[Pasted image 20260228165001.png]]
- Representation Model将当前的observation和之前的state和action映射为当前的state
- transition model: 预测未来state
- reward model: 预测state下的reward
Latent Prediction的优势：
- 内存占用低
- 高效并行
- 预测能力
所以这里的Repr Model中前一个状态，更像是一种RNN的回望，而不是Prediction，实现了上下文整合
### Learning Behaviors by Latent Imagination
Imagined Trajectories从观察过的states $s_t$中开始，并遵循world model中建模的transition与reward, action来进行交互
policy同样需要一个神经网络来进行建模
![[Pasted image 20260228181859.png]]Dynamics Learning利用表示模型计算对应潜状态
Behaviour Learning就是使用梦境训练动作模型和价值模型
Environment Interaction就是使用学习的动作模型，进行实际交互
#### Action and Value Models
Action and Value Model，action为了预测actions来解决imagine env，value model则是为了预测特定state的value
action输出的是tanh Gaussian，为了方便进行梯度回传
![[Pasted image 20260228183000.png]]
#### Value Estimation
先拟合一个n-step return，累加前k步的奖励
使用指数加权平均，将上述不同长度的k步返回值进行指数加权平均
![[Pasted image 20260228184029.png]]
#### Learning Objective
为了更新action与value model，首先计算value estimates
action model $q_\phi(a_\tau | s_\tau)$ 的目的就是预测更高value estimates的action
![[Pasted image 20260228185034.png]]
value model的更新目标就是回归targets
### Learning Latent Dynamics
关注latent dynamic models，在一个紧凑的latent space内进行前向的prediction，关注如何生成$s_t$
#### Reward Prediction
在有限的数据集中，学习与reward有关的observation是非常重要的
#### Reconstruction
这里指的是重建图片对应的Representation
![[Pasted image 20260301110146.png]]
使用RSSM进行建模，结合CNN进行image observation
这里的Loss结合了图像重建的损失，奖励预测的损失与KL散度的正则项(看到图像后的编码，与上一个时刻的预测)

核心的创新点：
- 解析梯度回传，不再依赖无梯度优化，使用Reparameterization
	- 连续动作使用Repara
	- 离散动作使用straight-through grad
- 在latent space中进行大规模的Imagination
- 引入Actor-Critic
- 利用$\lambda-target$平衡偏差与方差

# Mastering Diverse Domains through World Models
DreamerV3
### Introduction
现在的RL算法适应不同的环境
DreamerV3在不同的model sizes下和training budgets喜爱都可以完成训练，且不需要微调超参数
### Learning Algorithm
包含了三个主要的神经网络
- world model预测actions之后的状态
- critic
- actor
在replayed experience上训练，在diverse domains上使用fixed hyperparameters
#### World Model Learning
使用autoencoding编码sensory inputs
使用RSSM，确定性状态转移，并凭借历史预测当前可能的状态
![[Pasted image 20260301114800.png]]
这里左侧的图片说明RSSM的基本架构
将视觉输入转化为$z_t$，使用当前state$h_t$与$a_t$进行预测
首先使用Encoder将视觉输入$x_t$转化为$z_t$，接下来，一个sequence model，带有状态$h_t$与$a_{t-1}$，预计序列。
拼接$h_t$与$z_t$，作为model state，用来预计$r_t$等等
![[Pasted image 20260301115410.png]]
$h_t$就是Deterministic Recurrent State，负责记忆与上下文，总结了截止到上一时刻的所有动作和状态序列
$z_t$是通过$h_t$和$x_t$编码得到的当前状态标识，这里和World Models这篇初始论文是有所不同的。
Continue Predictor就是预测任务是否继续，如果是1才能继续任务
给定的训练数据就是$x_{1:T}, a_{1:T}, r_{1:T}, c_{1:T}$ 
优化过程是end-to-end的
![[Pasted image 20260301125943.png]]
使用的Loss风格各有不同
Prediction Loss是让reward predictor，continue predictor等等最小，确保Latent Space可以正确换元
Dynamics Loss是让predictor $z_t$与实际表达$z_t$之间的KL-Divergence最小
Representation Loss是训练Encoder，让Encoder能够总结出正确的规律
![[Pasted image 20260301131617.png]]
pred是正确预测图像，奖励与continuition
dyn是正确预测下一个状态的state
rep是让encoder生成易于学习的表示
可以注意到dyn和rep的Loss是反方向的，也就是训练的对象是不同的
#### Critic Learning
完全从world model预测的representations中学习
operate on model states $s_t = \{h_t, z_t\}$
Actor的目的是最大化输出的Discounted Returns
计算value同样使用的是bootstrapped $\lambda$-returns
![[Pasted image 20260301133403.png]]
预测的是return的概率分布
转化为Categorical Distribution
#### Actor Learning
使用不同percentile的return，以及使用Exponential Moving Average来平滑return
![[Pasted image 20260301144247.png]]
使用了这里的Reinforce Estimator
#### Robust Predictions
large targets可能难以预测，所以使用symlog函数来作为Loss解决问题
![[Pasted image 20260301135335.png]]
零点附近是对称的
Loss使用two-hot targets作为targets，twohot向量就是一个向量中其他位置都是0，只有k和k+1是非0值，且这两个加起来是1
训练的目标是最小化Cross Entropy Loss
所以这一部分的主要内容就是，首先是引入symlog函数对预测目标进行变换，其次twohot解决对于随机目标的回归问题。

==Analytic Gradient Path?==
- 世界模型本身是Differentiable的，说明当前的动作$a_t$会影响未来的状态，进而影响后续的其他奖励
==Straight Through Gradients?==
# Training Agents Inside of Scalable World Models
Dreamer4
之前的World Model存在无法精准预测Object Interactions的问题
能够完全从offline data中学习获得diamond
world model的核心问题：预测action之后的future outcomes
imagination中进行RL
可以发现现在的world model能够准确预测object interactions与game mechanics
基于Flow Matching建立
Shortcut Models是一种改进的Diffusion/Flow Matching Model
- 感知信号水平与请求步长
- 使用较少的采样步进行学习
Diffusion Forcing:
- 将Diffusion Model用于序列数据
- 为序列中的每一个时间步分配不同的信号水平，生成被污染的序列
- 当前时间步是去噪任务目标，也是后续生成的上下文，缓解误差累积
信号水平$\tau$刻画的是数据点的清晰程度，0代表纯噪声，1代表干净数据