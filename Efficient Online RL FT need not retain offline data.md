- ==关注讲故事+实验==
Pre-training是ML中非常常见的，一般是Pre-training on diverse data + **Task-Specific** fine-tuning
当下有很多RL fine-tuning methods会要求继续在offline data上进行训练
我们证明retaining offline data是不必要的
通常直接offline->online可能会导致Sudden Divergence in the value function
我们提出WSRL(Warm-Start RL)，添加一个warmup phase，帮助Q函数的Recalibration
# 1. Introduction
之前的方法往往需要Continued Training on Offline Data，可能存在的问题如下：
- offline dataset的规模会不断变大，offline dataset上的训练效率低下且昂贵
- 这一个过程让前面的offline pre-training失去意义，毕竟效果还不一定比得上online from scratch的方法
我们的目标就是构建一个Does not retain offline data的online RL approach
当前的pessimistic与behavioral constraint方法都会经历一个***Recalibration Phase*** ，value发生重大的改变
![[Pasted image 20250830113447.png]]
***Is it possible to fine-tune from offline RL value and policy initializations, but without retaining offline
data and not forget the pre-training?***
我们的思路是，使用一小部分的数据来模拟offline data retention，来帮助recalibration
在recalibration完成之后，我们可以运行最有效的online RL方法，进行最佳的online RL
我们先对online replay buffer进行初始化，后面就可以开始online RL
我们发现WSRL的学习速度更快，渐进性能也更好。
initialize with a small number of transitions from the pre-trained offline RL policy
# 2. Problem Formulation
我们的目标是只使用$\pi_\phi^{pre}以及Q_\theta^{pre}$ ，不使用$D_{off}$
# 3. Understanding the Role of Offline Data in Online FT
![[Pasted image 20250830121703.png]]
这些算法中，保留offline data都会有更好的效果。
## 3.1 The Role of offline data at the beginning of FT
IQL与CQL如果没有retain offline data，则会性能严重下降且难以恢复
Unlearning：训练初期的性能下滑
Forgetting：Pre-trained initialization的完全失去
Unlearning往往是难以避免的，因为Distribution Shift必然存在。
Forgetting是灾难性的，因为这样pre-training就一点用都没有了。
### Why does not retaining offline data hurt
![[Pasted image 20250830123634.png]]
在retained offline data减少的时候，我们的Q值会开始Diverge，这会反过来导致Divergence in the TD-Error，指示出了Forgetting现象的发生
但是这个TD-Error只在offline数据集上非常高，而在online上并不高
在fine-tuning阶段使用offline data进行训练会影响offline data的拟合
![[Pasted image 20250830124459.png]]Q-values会出现Distribution Diverge
### Why are Q underestimated
offline data上的Q会发散，online上的Q同样会发散。
接下来理解为什么Recalibration会导致Q值的发散。
因为offline RL pre-training是conservative的，所以learned Q-values期望很低
![[Pasted image 20250830153051.png]]
因为我们的Q-target也会变小，所以会导致一个Downward Spiral让Q-function的值一直都在变小。
![[Pasted image 20250830153339.png]]
Q-value recalibration可能会导致严重的Underestimation
### Conclusion
如果不保留离线数据，Q-Function在offline数据上的拟合会崩溃，导致offline下发散，online上被低估。
因为本身就有保守性正则，这些OOD动作的Q值容易被压得过低，导致了下行螺旋。
如果保留offline data，我们batch中有相当一部分是熟悉的transition，正则项不会进行过度压低。
## 3.2 The Adverse Impact of Offline Data on Asymptotic Performance
continued training on offline data是会影响我们算法的性能与效率的。
![[Pasted image 20250830155306.png]]
可以看出retaining offline data还不如直接开始online RL，渐进性能差很多
![[Pasted image 20250830155845.png]]
# 4. WSRL: Fast FT without Offline Data Retention
现在我们的处境是：
- 保留offline data会延缓online FT
- 丢弃offline data会导致forgetting
### Key Idea
也许可以在FT阶段使用standard online RL
我们使用的是high UTD regime
但是如果我们不想catastrophic forgetting，就要simulate continued training
UTD较高的时候，更新batch size小，但是更新频率高，可以做掉较快进行数据拟合。
UTD就是每一个样本所对应的网络更新次数,UTD越高，对数据的利用率也就更高，也会更加稳定，但是容易导致过拟合。
### WSRL algorithm
先有offline阶段训练好的$Q_\theta^{pre}$与policy $\pi_\psi^{pre}$ 
接下来我们进行K online steps，使用的是原来的offline policy来模拟retention of offline data
在收集完数据之后，对value与policy进行训练。
接下来在FT阶段使用high UTD ratio
high UTD中的一些问题，使用Q-ensemble以及layer normalization等手段进行更新。
### Implementation Details
我们一般使用Cal-QL来进行initialization，online使用SAC
# 5. Experimental Evaluation
## 5.2 Can WSRL enable efficient ft in no-retention ft?
![[Pasted image 20250830162920.png]]
在no-retention setting中我们的方法可以说是遥遥领先
虽然WSRL也会有initial unlearning，但是恢复非常快。
我们的WSRL的Q-function也没有出现diverge的情况
### 5.3 Compare with methods that retain offline data
比如Cal-QL
我们的WSRL在使用相同的HIGH UTD以及Q-Ensemble的情况下仍然表现最好，渐进也快
![[Pasted image 20250830164223.png]]
## 5.4 How Critical is the warmup phase
去除warmup，可以观测到显著的性能下滑
![[Pasted image 20250830164301.png]]
我们发现这样的Warmup方式比直接使用offline dataset会更有效，这是因为我们防止了Divergence与Downward Spiral
在有warmup的时候，downward spiral被有效抑制了。
![[Pasted image 20250830164847.png]]
可以看出这样就避免了over-pessimistic的value
## 5.5 How important is using a standard online RL for FT
对比使用offline RL的算法，可以发现使用standard online RL算法的效果会明显更好。
看这张图的右侧
![[Pasted image 20250830165001.png]]
## 5.6 Initialize the policy/value function
policy initialization
- pre-trained的policy已经可以与环境之间进行交互了，我们可以发现有initialized的policy的效果明显更好。
- ![[Pasted image 20250830165158.png]]
- 这样的方法可以speed up online learning
Benefits of Q-initialization
- ![[Pasted image 20250830165001.png]]
- 看这张图的左侧，可以让fine-tuning加速
## 5.7 How does WSRL perform on real world robotics tasks
相比较于SERL与RLPD，都是先有pre-training，我们的WSRL可以在real-time RL training之后从offline performance提升到满分的性能。
SERL算法就会卡在一些ood positions上，不能够继续前进。
SERL经常停住不动，展现出对offline dataset的学习能力不够强
# 6. Related Works
o2o RL
- 使用offline dataset来高效率运行online RL
- 需要保留offline data
bottlenecks in online RL Fine-tuning
- 使用offline data虽然可以稳定训练，但是会导致训练速度变慢。
- 不保留offline data的时候，state-action shift是导致问题的原因
没有pre-training的onlineRL
- 直接让online RL从包含了离线数据的dataset开始学习，这种方法也相当有效
no retain的RL policy fine-tuning
- 许多持续学习与终身学习方法也会不保留先前经验并微调策略。
- 本文关注单环境单任务。
# 7. Conclusion
这里我们探索的是一种不用保留先前数据的方法，我们发现之前的算法因为Q-Divergence与Distribution Shift而经常失败，所以这里我们使用Warmup Phase来预防Q-divergence。