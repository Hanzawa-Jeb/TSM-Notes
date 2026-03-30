# 0. 我们的想法
## 0.1 基本解释
- offline to online RL的过程之中，我们==存在Distributional Shift==
- 这样的Distributional Shift，根据之前讨论Plasticity的文章的相关意见，可能会导致我们的性能下降
- 学习能力变弱，可能是FAU的下降导致的
- 所以我们是否可以通过解决FAU问题的一些方法，例如ReDo等等，来解决我们offline to online RL中存在的学习能力的问题？
## 0.2 实验方案
- 我们是否可以先验证在offline to online setting中的FAU下降问题？
- 复现Cal-QL论文中提到的哪些算法
# 1. 第一阶段：找到Uncalibrated Case中的FAU Tendency
## 1.1.实验目的
我们实验的目的是监测在offline to online这个过程中出现的FAU这个参量的变化。
我们的期望是：
- 没有Calibrated的QL算法：
	- 在offline to online的瞬间出现FAU==变化会是什么样的？==
	- Q均值上升
	- 但是我们的d4rl_normalized_score下降
## 1.2. 实验的组织方法
### 1.2.1 第一部分大致思路
- 我们要找到Cal-QL论文中提到的其他方法以及Cal-QL，复现Performance Drop的现象
- 观测在Performance Drop的同时FAU发生的变化
#### 两类算法
CQL：
- 可能出现突变，我们要检查相关参数的变化
IQL/...：
- 较差的Asymptotic Performance
- 所以我们想要观察到：
	- 较差的渐进性能
	- FAU的大幅度下降
### 1.2.2 论文中的知识支撑
- ![[Pasted image 20250910200103.png]]
- 之前的算法可以分为两类：IQL类与CQL类
	- 一类是Poor Asymptotic Performance，==例如IQL==
	- 一类是出现Performance Drop(Recalibration)，==例如传统CQL==
![[Pasted image 20250910200443.png]]
TD3+BC, IQL, AWAC出现了slow asymptotic performance的问题
CQL出现了Unlearning

## 1.3. 实验
### 1.3.1 实验中的基本参数设定要点
- Normalized Score的定义
![[Pasted image 20250910201740.png]]
### Un-calibrated QL
阅读Cal-QL论文的相关知识：
#### IQL/...(Poor Asymptotic Performance)
==我们想要找到的就是Poor Asymptotic Performance==
==然后找到这一过程是不是因为==

#### CQL(No Data for now)
==?==
==出现现象我们再来解释==
### Calibrated QL
#### HalfCheetah
##### HalfCheetah-Medium
![[Pasted image 20250910192628.png]]
上图是我们的normalized_score的计算
![[Pasted image 20250910192702.png]]
这里是我们的FAU情况
小总结：online阶段的一些网络出现了FAU的下降，但是之后的Normalized Score稳定维持在较高的位置。
但是这里好像看不出FAU造成的其他问题
#### Dataset Quality
# 2. 找到其他方法中的FAU问题
## 2.1 大致思路
- 我当前的想法是，论文中提到IQL等等的Asymptotic Performance较差
- 我们是否可以验证IQL等算法在Distributional Shift中出现的FAU Drop？
# Appendix：实验中的注意事项与参数设置
d4rl_normalized_score貌似都是直接通过d4rl的环境接口得来的
```python
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    successes = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        goal_achieved = False
        while not done:
            action = actor.act(state, device)
            state, reward, done, env_infos = env.step(action)
            episode_reward += reward
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
        # Valid only for environments with goal
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward)
  
    actor.train()
    return np.asarray(episode_rewards), np.mean(successes)
```
success rate是这样进行统计的，也就是我们在无梯度的过程中进行n_episodes次测试，看成功了多少次，随后进行求平均值
![[Pasted image 20250911114236.png]]
D4RL官方给出的定义
# memo
找到出现 Performance Collapse的复现环境->具体参数
再找一下复现的
尽量不在Adroit上做验证
在现象中找到FAU的变化规律
找一下o2o中的原始settings
## 复现
![[Pasted image 20250911164427.png]]
这里是论文中提到的，这一个initial unlearning appears in multiple tasks as we show in Appendix F
![[Pasted image 20250911164737.png]]
这里是我们Appendix F中的内容：
![[Pasted image 20250911164901.png]]
可以发现在Franka Kitchen中的initial unlearning较为明显
但是在High Coverage of data的环境中这一现象并不明显