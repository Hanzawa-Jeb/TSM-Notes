# Cal-QL(DDP) FAU-Monitor
如果显存占用很少，可以增大Batch Size，但是在增大Batch Size的时候要以相同的比例增大我们的Learning Rate，这个就是Linear Rule
我们在进行数据传输的时候尽量直接让SSH连接建立在Server与Powershell之间，因为WSL的代理设置会有一些问题。SCP命令与SSH命令都直接建立在powershell与server之间即可。
使用tmux进行线程管理，否则进程会在当前SSH Session结束后停止运行
==不过请注意我们的batch_size指的是单卡中的batch_size，两次实验都是==
## Workflow of training
- 使用**Powershell**进行SSH连接
- 激活tmux终端
- 激活conda venv
- export WANDB_DIR为本次wandb本地目录保存地
- export WANDB_MODE=offline来规避需要实时上传的代理问题
- 在tmux终端中开始运行
# Code & Experiment Result
## Cal-QL-DDP
默认采用的是`halfcheetah-medium-expert-v2`
采用的是相似的FAU跟踪方法
解读一下相关参数
![[Pasted image 20250909154603.png]]
这里是两个critic的FAU变化情况
![[Pasted image 20250909154826.png]]
这里是actor三层的FAU变化情况
## IQL-DDP
当前我们默认的环境是`antmaze-umaze-v2`
我们定义一个class `FAU_Tracker`来追踪过程中的FAU变化情况
==请注意我们的训练过程是offline 1M, online 1M==
```python
class FAUTracker:
    """Tracks Fraction of Active Units (FAU) for neural network modules"""
    def __init__(self):
        self.activations = {}
        self.hooks = []
    def register_hooks(self, model, prefix=""):
        """Register forward hooks to track activations"""
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                hook_name = f"{prefix}_{name}" if prefix else name
                hook = module.register_forward_hook(
                    lambda module, input, output, hook_name=hook_name:
                    self._hook_fn(hook_name, output)
                )
                self.hooks.append(hook)
    def _hook_fn(self, name, activation):
        """Hook function to store activations"""
        if isinstance(activation, torch.Tensor):
            # Store the activation for later FAU computation
            self.activations[name] = activation.detach().clone()
    def compute_fau(self):
        """Compute FAU for all tracked modules"""
        fau_dict = {}
        for name, activation in self.activations.items():
            # FAU = sum(activation > 0) / total_neurons
            active_units = (activation > 0).float().mean()
            fau_dict[f"fau/{name}"] = active_units.item()
        # Clear activations after computation
        self.activations.clear()
        return fau_dict
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
```
我们这里的`compute_fau()`操作就是通过计算activation在全体unit中的占比
解读一下我们正在监控的fau参数：blabla.1， blabla.3的意思就是在我们的MLP中的第1层和第3层，因为FAU都是在activation function这一层进行监控的，这就是这两层对应的FAU情况
q1, q2就是我们的TwinQ中的两个Q函数
![[Pasted image 20250909105013.png]]
这里是value network的第一层和第三层的FAU
![[Pasted image 20250909105052.png]]
这里是critic网络第一层和第三层的FAU
![[Pasted image 20250909105222.png]]
这里是actor_net的FAU
# 关于性能评估如何进行
```python
@torch.no_grad()
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
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward)
  
    actor.train()
    return np.asarray(episode_rewards), np.mean(successes)
```
