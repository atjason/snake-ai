import math

import gym
import numpy as np

from snake_game import SnakeGame

class SnakeEnv(gym.Env): # 创建一个SnakeEnv类，继承自gym.Env
    def __init__(self, seed=0, board_size=12, silent_mode=True, limit_step=True):
        super().__init__() # 调用父类gym.Env的初始化方法
        self.game = SnakeGame(seed=seed, board_size=board_size, silent_mode=silent_mode) # 创建一个SnakeGame实例
        self.game.reset() # 重置游戏

        self.silent_mode = silent_mode # 设置silent_mode

        self.action_space = gym.spaces.Discrete(4) # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, # 设置observation_space
            shape=(84, 84, 3), # 设置observation_space的形状
            dtype=np.uint8 # 设置observation_space的数据类型
        )

        self.board_size = board_size # 设置board_size
        self.grid_size = board_size ** 2 # Max length of snake is board_size^2
        self.init_snake_size = len(self.game.snake)
        self.max_growth = self.grid_size - self.init_snake_size

        self.done = False # 设置done

        if limit_step: # 如果limit_step为True
            self.step_limit = self.grid_size * 4 # More than enough steps to get the food.
        else:
            self.step_limit = 1e9 # Basically no limit.
        self.reward_step_counter = 0

    def reset(self):
        self.game.reset() # 重置游戏

        self.done = False # 设置done
        self.reward_step_counter = 0 # 设置reward_step_counter

        obs = self._generate_observation() # 生成observation
        return obs
    
    def step(self, action):
        self.done, info = self.game.step(action) # info = {"snake_size": int, "snake_head_pos": np.array, "prev_snake_head_pos": np.array, "food_pos": np.array, "food_obtained": bool}
        obs = self._generate_observation() # 生成observation

        reward = 0.0 # 设置reward
        self.reward_step_counter += 1 # 增加reward_step_counter

        if info["snake_size"] == self.grid_size: # Snake fills up the entire board. Game over.
            reward = self.max_growth * 0.1 # Victory reward
            self.done = True # 设置done
            if not self.silent_mode: # 如果silent_mode为False
                self.game.sound_victory.play() # 播放胜利音效
            return obs, reward, self.done, info
        
        if self.reward_step_counter > self.step_limit: # Step limit reached, game over.
            self.reward_step_counter = 0 # 重置reward_step_counter
            self.done = True # 设置done
        
        if self.done: # Snake bumps into wall or itself. Episode is over.
            # Game Over penalty is based on snake size.
            reward = - math.pow(self.max_growth, (self.grid_size - info["snake_size"]) / self.max_growth) # (-max_growth, -1)            
            reward = reward * 0.1 # 设置reward
            return obs, reward, self.done, info
          
        elif info["food_obtained"]: # Food eaten. Reward boost on snake size.
            reward = info["snake_size"] / self.grid_size # 设置reward
            self.reward_step_counter = 0 # 重置reward_step_counter
        
        else:
            # Give a tiny reward/penalty to the agent based on whether it is heading towards the food or not.
            # Not competing with game over penalty or the food eaten reward.
            if np.linalg.norm(info["snake_head_pos"] - info["food_pos"]) < np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"]):
                reward = 1 / info["snake_size"] # 设置reward
            else:
                reward = - 1 / info["snake_size"] # 设置reward
            reward = reward * 0.1 # 设置reward

        # max_score: 72 + 14.1 = 86.1
        # min_score: -14.1

        return obs, reward, self.done, info
    
    def render(self):
        self.game.render() # 渲染游戏

    def get_action_mask(self): # 获取动作掩码
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])
    
    # Check if the action is against the current direction of the snake or is ending the game.
    def _check_action_validity(self, action): # 检查动作是否有效
        current_direction = self.game.direction # 获取当前方向
        snake_list = self.game.snake # 获取蛇列表
        row, col = snake_list[0] # 获取蛇头位置
        if action == 0: # UP
            if current_direction == "DOWN":
                return False
            else:
                row -= 1

        elif action == 1: # LEFT
            if current_direction == "RIGHT":
                return False
            else:
                col -= 1

        elif action == 2: # RIGHT 
            if current_direction == "LEFT":
                return False
            else:
                col += 1     
        
        elif action == 3: # DOWN 
            if current_direction == "UP":
                return False
            else:
                row += 1

        # Check if snake collided with itself or the wall. Note that the tail of the snake would be poped if the snake did not eat food in the current step.
        if (row, col) == self.game.food: # 如果蛇头位置等于食物位置
            game_over = (
                (row, col) in snake_list # The snake won't pop the last cell if it ate food.
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )
        else:
            game_over = (
                (row, col) in snake_list[:-1] # The snake will pop the last cell if it did not eat food.
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )

        if game_over:
            return False
        else:
            return True

    # EMPTY: BLACK; SnakeBODY: GRAY; SnakeHEAD: GREEN; FOOD: RED;
    def _generate_observation(self): # 生成observation
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.uint8) # 创建一个全0的矩阵

        # Set the snake body to gray with linearly decreasing intensity from head to tail.
        obs[tuple(np.transpose(self.game.snake))] = np.linspace(200, 50, len(self.game.snake), dtype=np.uint8) # 设置蛇身颜色
        
        # Stack single layer into 3-channel-image.
        obs = np.stack((obs, obs, obs), axis=-1) # 将单层堆叠成3通道图像
        
        # Set the snake head to green and the tail to blue
        obs[tuple(self.game.snake[0])] = [0, 255, 0] # 设置蛇头颜色
        obs[tuple(self.game.snake[-1])] = [255, 0, 0] # 设置蛇尾颜色

        # Set the food to red
        obs[self.game.food] = [0, 0, 255] # 设置食物颜色

        # Enlarge the observation to 84x84
        obs = np.repeat(np.repeat(obs, 7, axis=0), 7, axis=1) # 将图像重复7次

        return obs

# Test the environment using random actions
# NUM_EPISODES = 100
# RENDER_DELAY = 0.001
# from matplotlib import pyplot as plt

# if __name__ == "__main__":
#     env = SnakeEnv(silent_mode=False)
    
    # # Test Init Efficiency
    # print(MODEL_PATH_S)
    # print(MODEL_PATH_L)
    # num_success = 0
    # for i in range(NUM_EPISODES):
    #     num_success += env.reset()
    # print(f"Success rate: {num_success/NUM_EPISODES}")

    # sum_reward = 0

    # # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
    # action_list = [1, 1, 1, 0, 0, 0, 2, 2, 2, 3, 3, 3]
    
    # for _ in range(NUM_EPISODES):
    #     obs = env.reset()
    #     done = False
    #     i = 0
    #     while not done:
    #         plt.imshow(obs, interpolation='nearest')
    #         plt.show()
    #         action = env.action_space.sample()
    #         # action = action_list[i]
    #         i = (i + 1) % len(action_list)
    #         obs, reward, done, info = env.step(action)
    #         sum_reward += reward
    #         if np.absolute(reward) > 0.001:
    #             print(reward)
    #         env.render()
            
    #         time.sleep(RENDER_DELAY)
    #     # print(info["snake_length"])
    #     # print(info["food_pos"])
    #     # print(obs)
    #     print("sum_reward: %f" % sum_reward)
    #     print("episode done")
    #     # time.sleep(100)
    
    # env.close()
    # print("Average episode reward for random strategy: {}".format(sum_reward/NUM_EPISODES))
