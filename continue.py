import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0005  # Giảm learning rate để quá trình học ổn định hơn

class Agent:

    def __init__(self, model_path=None):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.95  # Tăng gamma để ưu tiên phần thưởng dài hạn
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, 512, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.epsilon_min = 0.01  # Giá trị epsilon tối thiểu
        self.epsilon_decay = 0.995  # Hệ số giảm epsilon

        # Tải trọng số mô hình nếu có đường dẫn
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        self.model.load(model_path)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # Sử dụng epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        exploration_rate = self.epsilon * (1 - self.n_games / 500)
        final_move = [0, 0, 0]
        if random.uniform(0, 1) < exploration_rate:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


average_score_window = 100
no_improvement_count = 0
max_no_improvement = 30  # Dừng nếu không có cải thiện trong 10 lần


def train(model_path=None):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(model_path)  
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Điều chỉnh phần thưởng ưu tiên sinh tồn hơn
        if done:
            reward = -10  # Phạt nặng khi rắn chết
        elif reward > 0:
            reward = 10  # Phần thưởng nhỏ khi ăn táo
        else:
            reward = 0.5  # Phần thưởng khi sống sót qua lượt di chuyển

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()  # Lưu mô hình nếu đạt kỷ lục mới

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            # Dừng nếu không có cải thiện điểm
            if agent.n_games > average_score_window:
                if mean_score <= record:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0

                if no_improvement_count >= max_no_improvement:
                    print("Training stopped after", agent.n_games, "games without improvement.")
                    break
            
            plot(plot_scores, plot_mean_scores)

            # Dừng dựa trên số lượt trò chơi
            if agent.n_games >= 500:
                print("Training complete after", agent.n_games, "games.")
                break


if __name__ == '__main__':
    # Đường dẫn đến mô hình đã huấn luyện trước đó để tiếp tục huấn luyện
    train(model_path='model\model_3.pth')
