import os
import sys
import random

import numpy as np

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
from pygame import mixer

class SnakeGame:
    def __init__(self, seed=0, board_size=12, silent_mode=True): # 初始化游戏
        self.board_size = board_size # 设置board_size
        self.grid_size = self.board_size ** 2 # 设置grid_size
        self.cell_size = 40 # 设置cell_size
        self.width = self.height = self.board_size * self.cell_size # 设置width和height

        self.border_size = 20 # 设置border_size
        self.display_width = self.width + 2 * self.border_size # 设置display_width
        self.display_height = self.height + 2 * self.border_size + 40 # 设置display_height

        self.silent_mode = silent_mode # 设置silent_mode
        if not silent_mode: # 如果silent_mode为False
            pygame.init() # 初始化pygame
            pygame.display.set_caption("Snake Game") # 设置游戏标题
            self.screen = pygame.display.set_mode((self.display_width, self.display_height)) # 设置屏幕
            self.font = pygame.font.Font(None, 36) # 设置字体

            # Load sound effects
            mixer.init()
            self.sound_eat = mixer.Sound("sound/eat.wav")
            self.sound_game_over = mixer.Sound("sound/game_over.wav")
            self.sound_victory = mixer.Sound("sound/victory.wav")
        else:
            self.screen = None
            self.font = None

        self.snake = None
        self.non_snake = None

        self.direction = None
        self.score = 0
        self.food = None
        self.seed_value = seed

        random.seed(seed) # Set random seed.
        
        self.reset()

    def reset(self): # 重置游戏
        self.snake = [(self.board_size // 2 + i, self.board_size // 2) for i in range(1, -2, -1)] # Initialize the snake with three cells in (row, column) format.
        self.non_snake = set([(row, col) for row in range(self.board_size) for col in range(self.board_size) if (row, col) not in self.snake]) # Initialize the non-snake cells.
        self.direction = "DOWN" # 蛇向下开始
        self.food = self._generate_food()
        self.score = 0

    def step(self, action): # 执行动作
        self._update_direction(action) # 更新方向

        # Move snake based on current action.
        row, col = self.snake[0] # 获取蛇头位置
        if self.direction == "UP": # 如果方向为向上
            row -= 1
        elif self.direction == "DOWN": # 如果方向为向下
            row += 1
        elif self.direction == "LEFT": # 如果方向为向左
            col -= 1
        elif self.direction == "RIGHT": # 如果方向为向右
            col += 1

        # Check if snake eats food.
        if (row, col) == self.food: # 如果蛇头位置等于食物位置
            food_obtained = True # 食物被吃
            self.score += 10 # 加10分
            if not self.silent_mode: # 如果silent_mode为False
                self.sound_eat.play() # 播放吃食物的声音
        else:
            food_obtained = False # 食物未被吃
            self.non_snake.add(self.snake.pop()) # 弹出蛇的最后一个细胞并将其添加到非蛇集合中

        # 检查蛇是否与自身或墙壁碰撞
        done = (
            (row, col) in self.snake # 蛇头位置在蛇身上
            or row < 0 # 蛇头位置在墙壁上
            or row >= self.board_size # 蛇头位置在墙壁上
            or col < 0 # 蛇头位置在墙壁上
            or col >= self.board_size # 蛇头位置在墙壁上
        )

        if not done:
            self.snake.insert(0, (row, col)) # 将蛇头位置插入到蛇身上
            self.non_snake.remove((row, col)) # 从非蛇集合中移除蛇头位置

        else: # 如果游戏结束且游戏不处于静默模式
            if not self.silent_mode: # 如果silent_mode为False
                if len(self.snake) < self.grid_size: # 如果蛇的长度小于grid_size
                    self.sound_game_over.play() # 播放游戏结束的声音
                else:
                    self.sound_victory.play() # 播放胜利的声音

        # 在蛇移动完成后添加新的食物
        if food_obtained:
            self.food = self._generate_food() # 生成新的食物

        info = {
            "snake_size": len(self.snake), # 蛇的长度
            "snake_head_pos": np.array(self.snake[0]), # 蛇头位置
            "prev_snake_head_pos": np.array(self.snake[1]), # 蛇头位置
            "food_pos": np.array(self.food), # 食物位置
            "food_obtained": food_obtained # 食物是否被吃
        }

        return done, info

    # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
    def _update_direction(self, action):
        if action == 0:
            if self.direction != "DOWN":
                self.direction = "UP"
        elif action == 1:
            if self.direction != "RIGHT":
                self.direction = "LEFT"
        elif action == 2:
            if self.direction != "LEFT":
                self.direction = "RIGHT"
        elif action == 3:
            if self.direction != "UP":
                self.direction = "DOWN"
        # Swich Case is supported in Python 3.10+

    def _generate_food(self):
        if len(self.non_snake) > 0: # 如果非蛇集合不为空
            food = random.sample(self.non_snake, 1)[0] # 从非蛇集合中随机选择一个位置作为食物
        else: # 如果蛇占据了整个棋盘，则不需要生成新的食物，直接默认返回(0, 0)
            food = (0, 0)
        return food
    
    def draw_score(self):
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255)) # 渲染得分文本
        self.screen.blit(score_text, (self.border_size, self.height + 2 * self.border_size)) # 在屏幕上绘制得分文本
    
    def draw_welcome_screen(self):
        title_text = self.font.render("SNAKE GAME", True, (255, 255, 255)) # 渲染标题文本
        start_button_text = "START" # 渲染开始按钮文本

        self.screen.fill((0, 0, 0)) # 填充屏幕
        self.screen.blit(title_text, (self.display_width // 2 - title_text.get_width() // 2, self.display_height // 4)) # 在屏幕上绘制标题文本
        self.draw_button_text(start_button_text, (self.display_width // 2, self.display_height // 2)) # 在屏幕上绘制开始按钮文本
        pygame.display.flip() # 刷新屏幕

    def draw_game_over_screen(self):
        game_over_text = self.font.render("GAME OVER", True, (255, 255, 255)) # 渲染游戏结束文本
        final_score_text = self.font.render(f"SCORE: {self.score}", True, (255, 255, 255)) # 渲染最终得分文本
        retry_button_text = "RETRY" # 渲染重试按钮文本

        self.screen.fill((0, 0, 0)) # 填充屏幕
        self.screen.blit(game_over_text, (self.display_width // 2 - game_over_text.get_width() // 2, self.display_height // 4)) # 在屏幕上绘制游戏结束文本
        self.screen.blit(final_score_text, (self.display_width // 2 - final_score_text.get_width() // 2, self.display_height // 4 + final_score_text.get_height() + 10)) # 在屏幕上绘制最终得分文本
        self.draw_button_text(retry_button_text, (self.display_width // 2, self.display_height // 2)) # 在屏幕上绘制重试按钮文本
        pygame.display.flip() # 刷新屏幕

    def draw_button_text(self, button_text_str, pos, hover_color=(255, 255, 255), normal_color=(100, 100, 100)):
        mouse_pos = pygame.mouse.get_pos() # 获取鼠标位置
        button_text = self.font.render(button_text_str, True, normal_color) # 渲染按钮文本
        text_rect = button_text.get_rect(center=pos) # 获取按钮文本的矩形
        
        if text_rect.collidepoint(mouse_pos): # 如果鼠标位置在按钮文本上
            colored_text = self.font.render(button_text_str, True, hover_color) # 渲染高亮文本
        else:
            colored_text = self.font.render(button_text_str, True, normal_color) # 渲染普通文本
        
        self.screen.blit(colored_text, text_rect) # 在屏幕上绘制文本
    
    def draw_countdown(self, number):
        countdown_text = self.font.render(str(number), True, (255, 255, 255)) # 渲染倒计时文本
        self.screen.blit(countdown_text, (self.display_width // 2 - countdown_text.get_width() // 2, self.display_height // 2 - countdown_text.get_height() // 2)) # 在屏幕上绘制倒计时文本
        pygame.display.flip() # 刷新屏幕

    def is_mouse_on_button(self, button_text):
        mouse_pos = pygame.mouse.get_pos() # 获取鼠标位置
        text_rect = button_text.get_rect(
            center=(
                self.display_width // 2,
                self.display_height // 2,
            )
        )
        return text_rect.collidepoint(mouse_pos)

    def render(self):
        self.screen.fill((0, 0, 0)) # 填充屏幕

        # Draw border
        pygame.draw.rect(self.screen, (255, 255, 255), (self.border_size - 2, self.border_size - 2, self.width + 4, self.height + 4), 2) # 绘制边框

        # Draw snake
        self.draw_snake() # 绘制蛇
        
        # Draw food
        if len(self.snake) < self.grid_size: # If the snake occupies the entire board, don't draw food.
            r, c = self.food
            pygame.draw.rect(self.screen, (255, 0, 0), (c * self.cell_size + self.border_size, r * self.cell_size + self.border_size, self.cell_size, self.cell_size)) # 绘制食物

        # Draw score
        self.draw_score() # 绘制得分

        pygame.display.flip() # 刷新屏幕

        for event in pygame.event.get(): # 获取事件
            if event.type == pygame.QUIT: # 如果事件类型为退出
                pygame.quit() # 退出pygame
                sys.exit() # 退出程序

    def draw_snake(self):
        # Draw the head
        head_r, head_c = self.snake[0] # 获取蛇头位置
        head_x = head_c * self.cell_size + self.border_size # 计算蛇头x坐标
        head_y = head_r * self.cell_size + self.border_size # 计算蛇头y坐标

        # Draw the head (Blue)
        pygame.draw.polygon(self.screen, (100, 100, 255), [ 
            (head_x + self.cell_size // 2, head_y), # 绘制蛇头
            (head_x + self.cell_size, head_y + self.cell_size // 2), # 绘制蛇头
            (head_x + self.cell_size // 2, head_y + self.cell_size), # 绘制蛇头
            (head_x, head_y + self.cell_size // 2) # 绘制蛇头
        ])

        eye_size = 3 # 眼睛大小
        eye_offset = self.cell_size // 4 # 眼睛偏移量
        pygame.draw.circle(self.screen, (255, 255, 255), (head_x + eye_offset, head_y + eye_offset), eye_size) # 绘制眼睛
        pygame.draw.circle(self.screen, (255, 255, 255), (head_x + self.cell_size - eye_offset, head_y + eye_offset), eye_size) # 绘制眼睛

        # Draw the body (color gradient)
        color_list = np.linspace(255, 100, len(self.snake), dtype=np.uint8) # 颜色列表
        i = 1 # 初始化i
        for r, c in self.snake[1:]: # 遍历蛇身
            body_x = c * self.cell_size + self.border_size # 计算蛇身x坐标
            body_y = r * self.cell_size + self.border_size # 计算蛇身y坐标
            body_width = self.cell_size # 蛇身宽度
            body_height = self.cell_size # 蛇身高度
            body_radius = 5 # 蛇身半径
            pygame.draw.rect(self.screen, (0, color_list[i], 0), # 绘制蛇身
                            (body_x, body_y, body_width, body_height), border_radius=body_radius) # 绘制蛇身
            i += 1
        pygame.draw.rect(self.screen, (255, 100, 100), # 绘制蛇身
                            (body_x, body_y, body_width, body_height), border_radius=body_radius) # 绘制蛇身
        

if __name__ == "__main__":
    import time

    seed = random.randint(0, 1e9) # 随机种子
    game = SnakeGame(seed=seed, silent_mode=False) # 初始化游戏
    pygame.init() # 初始化pygame
    game.screen = pygame.display.set_mode((game.display_width, game.display_height)) # 设置屏幕
    pygame.display.set_caption("Snake Game") # 设置游戏标题
    game.font = pygame.font.Font(None, 36) # 设置字体
    

    game_state = "welcome" # 游戏状态

    # Two hidden button for start and retry click detection
    start_button = game.font.render("START", True, (0, 0, 0)) # 渲染开始按钮文本
    retry_button = game.font.render("RETRY", True, (0, 0, 0)) # 渲染重试按钮文本

    update_interval = 0.15 # 更新间隔
    start_time = time.time() # 开始时间
    action = -1 # 动作

    while True:
        
        for event in pygame.event.get(): # 获取事件

            if game_state == "running": # 如果游戏状态为运行中
                if event.type == pygame.KEYDOWN: # 如果事件类型为按键按下
                    if event.key == pygame.K_UP: # 如果按键为向上
                        action = 0 # 动作
                    elif event.key == pygame.K_DOWN: # 如果按键为向下
                        action = 3 # 动作
                    elif event.key == pygame.K_LEFT: # 如果按键为向左
                        action = 1 # 动作
                    elif event.key == pygame.K_RIGHT: # 如果按键为向右
                        action = 2 # 动作

            if event.type == pygame.QUIT: # 如果事件类型为退出
                pygame.quit() # 退出pygame
                sys.exit() # 退出程序

            if game_state == "welcome" and event.type == pygame.MOUSEBUTTONDOWN: # 如果游戏状态为欢迎界面且事件类型为鼠标按下
                if game.is_mouse_on_button(start_button): # 如果鼠标在开始按钮上
                    for i in range(3, 0, -1): # 倒计时
                        game.screen.fill((0, 0, 0)) # 填充屏幕
                        game.draw_countdown(i) # 绘制倒计时
                        game.sound_eat.play() # 播放吃食物的声音
                        pygame.time.wait(1000) # 等待1秒
                    action = -1  # 重置动作变量
                    game_state = "running" # 游戏状态为运行中

            if game_state == "game_over" and event.type == pygame.MOUSEBUTTONDOWN: # 如果游戏状态为游戏结束且事件类型为鼠标按下
                if game.is_mouse_on_button(retry_button): # 如果鼠标在重试按钮上
                    for i in range(3, 0, -1): # 倒计时
                        game.screen.fill((0, 0, 0)) # 填充屏幕
                        game.draw_countdown(i) # 绘制倒计时
                        game.sound_eat.play() # 播放吃食物的声音
                        pygame.time.wait(1000) # 等待1秒
                    game.reset() # 重置游戏
                    action = -1  # 重置动作变量
                    game_state = "running" # 游戏状态为运行中
        
        if game_state == "welcome": # 如果游戏状态为欢迎界面
            game.draw_welcome_screen() # 绘制欢迎界面

        if game_state == "game_over": # 如果游戏状态为游戏结束
            game.draw_game_over_screen() # 绘制游戏结束界面

        if game_state == "running": # 如果游戏状态为运行中
            if time.time() - start_time >= update_interval: # 如果时间间隔大于等于更新间隔
                done, _ = game.step(action) # 执行动作
                game.render() # 渲染游戏
                start_time = time.time() # 更新开始时间

                if done:
                    game_state = "game_over"
        
        pygame.time.wait(1)
