在IQL和CalQL上进行实验
在Actor蒸馏/Critic蒸馏/ActorCritic均蒸馏上探索可行性
在蒸馏中神经元活跃比例中探索可行性

在umaze_diverse上的表现不好，是否能通过降低distillation_steps来降低影响？或者使用layernorm?或者改变frozen的ratio?

![[Pasted image 20260324112742.png]]
之前的bug，CalQL是正常的
IQL之前在仅蒸馏Actor的时候有问题

All-Distill:
- no layernorm
Actor-Distill
- no layernorm
- layernorm

改变seed, frozen_ratio

是否能只蒸馏Actor，重新训练Critic，且在一定步骤上只更新Critic而不更新Actor?