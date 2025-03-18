import os
import sys
import random

import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from snake_game_custom_wrapper_cnn import SnakeEnv

if torch.backends.mps.is_available(): # 如果MPS可用
    NUM_ENV = 32 * 2 # 设置环境数量
else:
    NUM_ENV = 32 # 设置环境数量
LOG_DIR = "logs" # 设置日志文件夹

os.makedirs(LOG_DIR, exist_ok=True) # 创建日志文件夹

# 线性调度器
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str): # 如果initial_value是字符串
        initial_value = float(initial_value) # 转换为浮点数
        final_value = float(final_value) # 转换为浮点数
        assert (initial_value > 0.0) # 断言initial_value大于0

    def scheduler(progress): # 定义一个调度器
        return final_value + progress * (initial_value - final_value) # 返回最终值加上progress乘以initial_value减去final_value

    return scheduler

def make_env(seed=0): # 创建一个环境
    def _init(): # 初始化环境
        env = SnakeEnv(seed=seed) # 创建一个SnakeEnv环境
        env = ActionMasker(env, SnakeEnv.get_action_mask) # 使用ActionMasker包装环境
        env = Monitor(env) # 使用Monitor包装环境
        env.seed(seed) # 设置环境种子
        return env
    return _init

def main(): # 主函数

    # Generate a list of random seeds for each environment.
    seed_set = set() # 创建一个集合
    while len(seed_set) < NUM_ENV: # 当集合中的种子数量小于NUM_ENV时
        seed_set.add(random.randint(0, 1e9)) # 添加一个随机种子

    # Create the Snake environment.
    env = SubprocVecEnv([make_env(seed=s) for s in seed_set]) # 创建一个SubprocVecEnv环境

    if torch.backends.mps.is_available(): # 如果MPS可用
        lr_schedule = linear_schedule(5e-4, 2.5e-6) # 设置学习率调度器
        clip_range_schedule = linear_schedule(0.150, 0.025) # 设置clip范围调度器
        # Instantiate a PPO agent using MPS (Metal Performance Shaders).
        model = MaskablePPO(
            "CnnPolicy", # 使用CnnPolicy策略
            env, # 使用env环境
            device="mps", # 使用MPS设备
            verbose=1, # 设置verbose
            n_steps=2048, # 设置n_steps
            batch_size=512*8, # 设置batch_size
            n_epochs=4, # 设置n_epochs
            gamma=0.94, # 设置gamma
            learning_rate=lr_schedule, # 设置学习率
            clip_range=clip_range_schedule, # 设置clip范围
            tensorboard_log=LOG_DIR # 设置tensorboard日志
        )
    else:
        lr_schedule = linear_schedule(2.5e-4, 2.5e-6) # 设置学习率调度器
        clip_range_schedule = linear_schedule(0.150, 0.025) # 设置clip范围调度器
        # Instantiate a PPO agent using CUDA.
        model = MaskablePPO(
            "CnnPolicy", # 使用CnnPolicy策略
            env, # 使用env环境
            device="cuda", # 使用CUDA设备
            verbose=1, # 设置verbose
            n_steps=2048, # 设置n_steps
            batch_size=512, # 设置batch_size
            n_epochs=4, # 设置n_epochs
            gamma=0.94, # 设置gamma
            learning_rate=lr_schedule, # 设置学习率
            clip_range=clip_range_schedule, # 设置clip范围
            tensorboard_log=LOG_DIR # 设置tensorboard日志
        )

    # Set the save directory
    if torch.backends.mps.is_available():
        save_dir = "trained_models_cnn_mps" # 设置保存目录
    else:
        save_dir = "trained_models_cnn" # 设置保存目录
    os.makedirs(save_dir, exist_ok=True) # 创建保存目录

    checkpoint_interval = 15625 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_snake") # 创建一个CheckpointCallback

    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout # 保存原始stdout
    log_file_path = os.path.join(save_dir, "training_log.txt") # 设置日志文件路径
    with open(log_file_path, 'w') as log_file: # 打开日志文件
        sys.stdout = log_file # 将stdout重定向到日志文件

        model.learn(
            total_timesteps=int(100000000), # 设置总时间步
            callback=[checkpoint_callback] # 设置回调函数
        )
        env.close() # 关闭环境

    # Restore stdout
    sys.stdout = original_stdout # 恢复原始stdout

    # Save the final model
    model.save(os.path.join(save_dir, "ppo_snake_final.zip")) # 保存最终模型

if __name__ == "__main__":
    main() # 运行主函数
