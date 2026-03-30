## Abstract
为了解决Sample Efficiency与Exploration的问题，我们将offline data引入到online RL的问题当中。
我们的核心思路：
- 将off-policy methods也用来学习offline data
- 我们需要进行一系列比较小的改动来优化off-policy RL的性能。
不论是什么样的数据集都可以应用，无论是大量数据还是小量数据都可以。
# 1. Introduction
Online RL的强大性能一般都是基于与真实环境之间的online interaction的
真实环境中还可能会面临Sparse-Reward等问题
解决方案：使用Previous Policy生成的data或者human expert生成的数据
- 这样可以提供一个initial dataset, to "kick-start" 这个learning process
我们的方法需要
- 能够take advantage of offline data
- 能够减轻distribution shift可能带来的问题
我们的思路
- 没有offline pre-training与imitation terms
- 直接使用off-policy methods，用来学习offline data
我们实现方法能够成功的关键
- 创建 a minimal set of key design choices
- 首先提出一个symmetric sampling
- 防止value function的over-extrapolation
- 我们使用Layer Normalization来在一定程度上防止over-extrapolation
- Large Ensembles同样是非常重要的
RLPD(RL with Prior Data)
- 可以比先前的方法表现更好
- 仍然保留了online algorithms的主要优势
- 我们的方法同样具有Generality，在不同的offline datasets上表现良好
我们的方法证明了online off-policy方法在offline data上学习的强大性能
- 但是需要sampling方法
- normalizing
- large ensembles
这证明了我们每个individual ingredients的重要性。
且我们的模型在不同的offline data上（不管是expert还是sub-optimal）都有良好的表现。
# 2. Related Work
Offline RL pre-training
- 之前的方法同样参考使用了Large Ensembles以及Multiple Gradient Step的方法来增强数据的利用率
- 我们的方法介绍了Additional Hyperparameters
- 我们的方法==没有== offline pre-training
Constraining to prior data
- 有一种方法是让agent生成接近offline data分布的数据
- 我们也并没有使用BC Term限制我们的policy
- 我们的方法对**Dataset质量没有要求**
Unconstrained Methods with prior data
- 有一些方法在初始化replay buffer的时候使用了offline data
- 有一些方法平衡了online与offline数据的使用
- Balanced Sampling是非常重要的
# 3. Preliminaries
我们的方法是have access to offline datasets的
# 4. Online RL with Offline Data
我们的方法最好是对Pre-collected data的质量与数量agnostic的
我们的方法没有explicit constraints，也没有pre-training
基于SAC打造
我们的方法是minimally invasive的
## 4.1 Design Choice 1：How to incorporate offline data
我们的方法叫**Symmetric Sampling**
每一个batch
- 一半来自于Replay Buffer
- 另一半来自ofline data buffer
直接应用到canonical off-policy methods的时候，可能性能会不够好
## 4.2 Design Choice 2: Layer Normalization Mitigates Catastrophic Overestimation
![[Pasted image 20250902195316.png]]
传统的off-policy RL算法会对OOD actions不熟悉，出现overestimation，可能还会导致training instabilities and possible divergence
原本的方法可能是explicitly discourages OOD actions，可以被看做是anti-exploration
***我们的方法是，确保我们的funcion不要extrapolate***就可以了。
我们的Layer Normalization方法可以有效地Bound the Q-Values
LayerNorm可以做到限制Q值范数，防止发散的作用。
![[Pasted image 20250902195250.png]]
可以看出这里确实有效地抑制了Overestimation
## 4.3 Design Choice 3: Sample Efficient RL
我们对prior data的利用是根据online Bellman Backups而是隐性的
所以我们要尽可能地高效率地利用这些Bellman backups
一种方法是increase UTD(updates per environment step)
- 这可能会导致***更低的sample efficiency***
- 我们可以使用其他的方法，例如random ensemble distillation
- 我们的目的是要缓解over-fitting的问题
如何缓解over-fitting问题
- 使用Q-Ensemble防止overfitting问题
- 这里指的就是Random Ensemble Distillation这种方法。
## 4.4 Per-Environment Design Choices
deep RL算法往往是对Implementation Details非常敏感的，我们需要Per-Environment Hyperparameter Tuning
我们将会讨论这些参数是如何得来的
### Clipped Double Q-Learning(CDQ)
value-based methods往往会出现estimation uncertainty的问题，可能会导致value overestimation
CDQ就是采用两个Q网络中给出的结果更小的情况
![[Pasted image 20250903142849.png]]
这个方法可能表现出来过于Conservative了
### Maximum Entropy RL
我们在传统的return objective上加上一个entropy term
![[Pasted image 20250903143102.png]]
这里我们试图找到一个平衡robustness与exploration的方案
在maximize reward的同时behave as randomly as possible
这个可以在online fine-tuning阶段得到使用
- 因为这里的rewards往往是非常conservative的
- 且我们往往会需要更强的exploration性质
### Architecture
Network架构同样可以对于DeepRL的性能有很大的影响，不同的环境中的最优网络架构也有所不同
## 4.5 RLPD: Approach Overview
绿色为important approach，紫色为env-specific的design choices
![[Pasted image 20250903144759.png]]
集合LayerNorm, LargeEnsemble,
准备多个Critic网络以及对应的parameter
首先要initialize buffer $D$ with offline data
接下来将我们的在线交互放入$R$
接下来在每一次训练中从两边等量采样
使用Entropy+Q-Ensemble作为我们的target，minimize我们最终的loss，随后进行target network的更新，再更新我们的Policy
**EMA Weight**就是$\rho$，用来控制target network更新的速度。
# 5. Experiments
有几项对比需要完成：
- RLPD是否能够与有pre-training或者是having explicit constraints的算法之间对比
- 能不能转化到pixel-based env上
- LayerNorm是否能够减弱value divergence
### How does RLPD compare
Sparse Adroit与D4RL AntMaze都是Sparse Reward， D4RL Locomotion是dense reward
我们可以发现RLPD基本上可以与之前的SOTA表现得一样或者更好
### Does RLPD transfer to pixels
在only pixel observations的环境下，V-D4RL会更加困难
![[Pasted image 20250903151740.png]]
我们可以发现RLPD可以提供Consistent Improvements over purely online approaches
可以发现我们的RLPD方法可以有效帮助对于offline data的运用能力
我们经过实验发现，increasing UTD可以有效帮助性能的提升
## 5.1 RLPD Analysis and Ablation Study
### LayerNorm
我们可以发现LayerNorm被排除之后可能会导致更高的variance
我们的方法在最容易Diverge的情境下依然没有发生Divergence
去除掉LayerNorm后出现了collapsed performance
![[Pasted image 20250903152822.png]]
### Design Choice Worksflow
![[Pasted image 20250903152852.png]]
Entropy Backups 以及 smaller networks都会导致更差的performance
不同的方法适用于不同的环境问题
### Critic Regularization
![[Pasted image 20250903153111.png]]
比较三种方法，weight-decay, dropout与ensembling
我们可以发现Ensembling是最为有效的Regularization方法
### Buffer initialization
![[Pasted image 20250903154853.png]]
我们将Symmetric Sampling与initializing with offline data这种方法进行比较
我们可以发现symmetric sampling可以提升exploration性能
buffer initialization方法中虽然也同样有探索性，但是symmetric sampling的稳定性与方差表现更好，因为online data的数据可能方差会更高。
如果是中等质量的，且使用initializing the buffer的方法，渐进性能会更差。
### Sampling Proportion Sensitivity
我们评估对于Symmetric Ratio区别的影响
![[Pasted image 20250903155930.png]]
可以看出我们的算法对这个比例并不是很敏感
# 6. Conclusion
提出off-policy方法在online training上利用offline data的性能，我们发现RLPD的性能令人满意
- LayerNorm可以作为一个extrapolation regularizer
- sample efficient learning也非常关键
- 
- 我们的方法还对计算复杂度的要求影响非常小