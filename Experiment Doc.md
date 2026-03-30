![[Pasted image 20250903161102.png]]
我们做了一些FAU的监测工作
https://api.wandb.ai/links/hz_legoman-zhejiang-university/i6od5s37
我们尽量选择Locomotion任务，容易进行D4RL Normalized Score的评估，同时有三个数据集可以平衡不同方面的使用。
# Cal-QL, IQL FAU Tracking
Cal-QL的Plasticity上升现象不明显，反而在offline->online阶段存在FAU下降的问题
IQL因为采用的是Antmaze的Sparse Reward技术，导致我们的Normalized Score不稳定
# Seeking for FAU lift in offline to online setting
阅读论文Cal-QL
因为在offline to online这个setting中，如果没有Q-Recalibration的话，那么可能会出现Q-Overestimation导致的Degradation on Actor Performance
## 当前目标：试图复现这个Q-Recalibration中从offline到online阶段出现的Q-Recalibration与FAU变化
![[Pasted image 20250909181913.png]]这里是常见的一些D4RL Locomotion Task
接下来我们读一下Cal-QL的论文，试图找到这里找到Performance Drop的部分
![[Pasted image 20250909182329.png]]
这里就是我们的CQL的情况
![[Pasted image 20250909182947.png]]
我们对比多种算法
- TD3BC, IQL, AWAC主要面临的是==slow asymptotic performance==
- CQL主要面临的问题是==unlearn the offline initialization==
可能的思路：
- IQL等方法->Plasticity下降导致了Slow Asymptotic Performance
## CQL Experiment
![[Pasted image 20250909182558.png]]
这里是CQL在不同任务上出现的问题
![[Pasted image 20250909183519.png]]
这里是CQL采用的超参数
## Experiment
选择Adroit, Kitchen, Antmaze
### Adroit:
- alpha = 1
- mixing ratio = 0.5
#### Experiment 1
第一个实验我们使用的是如下Settings:
```yaml
# CQL with FAU Configuration for D4RL Adroit Environments
# This config implements the requested 1M offline + 1M online training with FAU tracking
  
# Environment settings
env: "relocate-human-v1"  # D4RL Adroit environment
device: "cuda:0"  # Single GPU training as requested
seed: 0
eval_seed: 0
  
# Training schedule (1M offline + 1M online as requested)
offline_steps: 1000000  # 1M offline steps
online_steps: 1000000   # 1M online steps
  
# Fixed hyperparameters as requested
alpha: 1.0              # Fixed CQL alpha parameter
mixing_ratio: 0.5       # Fixed mixing ratio for online phase
  
# CQL hyperparameters
batch_size: 256
discount: 0.99
alpha_multiplier: 1.0
use_automatic_entropy_tuning: true
backup_entropy: false
policy_lr: 3e-5
qf_lr: 3e-4
soft_target_update_rate: 5e-3
bc_steps: 0
target_update_period: 1
cql_alpha_online: 1.0   # Same as offline for consistency
cql_n_actions: 10
cql_importance_sample: true
cql_lagrange: false
cql_target_action_gap: -1.0
cql_temp: 1.0
cql_max_target_backup: false
cql_clip_diff_min: -1000000.0  # Effectively -inf
cql_clip_diff_max: 1000000.0   # Effectively +inf
orthogonal_init: true
normalize: true
normalize_reward: false
q_n_hidden_layers: 2
reward_scale: 1.0
reward_bias: 0.0
  
# Buffer settings
buffer_size: 2000000
  
# FAU tracking settings
track_fau: true
fau_threshold: 1e-6  # Threshold for active neuron detection
log_freq_steps: 1000  # Log FAU every 1000 steps
layer_include_patterns:
  - "base_network.*"  # Include policy network layers
  - "network.*"       # Include Q-function network layers
  
# Evaluation settings
eval_freq_steps: 5000  # Evaluate every 5k steps as requested
n_episodes: 10
  
# Checkpointing
checkpoints_path: "./checkpoints"
  
# WandB logging
project: "CQL-FAU-Adroit"
group: "CQL-D4RL-Adroit"
name: "CQL-FAU"
```
# 实验现象
## CQL
出现了CQL的Q-Explosion问题，Q值出现期望外的==突然上升==
