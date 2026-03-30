
# Reading List
IQL OK
SAC OK
Compare other baselines
PEX OK
Stay Hungry... OK
SARSA/DP... RL Basics 忘得差不多了: (
# Experiment Results
![[Pasted image 20260201131410.png]]
注意一下这里对比的这些Baseline，以及对比的环境是Adroit, MuJoCo和Antmaze
ICML25
# PEX
Policy Expansion中提到了在offline和Online阶段使用不同算法的思路
![[Pasted image 20260131140819.png]]
这里使用IQL作为offline阶段的算法，使用SAC作为online阶段的算法?
他这里将offline转换为online的方法：
使用两个Actor，一个是offline的，一个是online from scratch的，通过Critic通过Softmax进行采样。
关于online阶段Actor的风格是可以进行重点考虑的，可以使用IQL类似的Advantage-Weighted Behaviour Cloning的风格

# Paper Reading
## Stay Hungry, Keep Learning
使用基于KL-Divergence的思路来解决问题
## Reset&Distill
提出了Negative Transfer的概念，意思是Previous Knowledge from earlier tasks can become an obstacle to learning
所以应该要擦除之前learning过程中学到的Previous知识，同时也要防止Catastrophic Forgetting(因为这个过程是sequential的)，我们注意的是current task相关的内容。
提出了两个网络， online and offline learners
==???仍旧存疑，还是要再读一下==

