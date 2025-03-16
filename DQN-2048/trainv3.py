import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.table import Table

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2048 游戏环境类
class Game2048:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.add_random_tile()
        self.add_random_tile()

    def add_random_tile(self):
        empty_cells = np.argwhere(self.board == 0)
        if len(empty_cells) > 0:
            index = random.choice(empty_cells)
            self.board[index[0], index[1]] = 2 if random.random() < 0.9 else 4

    def move_left(self):
        reward = 0
        new_board = np.copy(self.board)
        for row in range(4):
            line = new_board[row]
            non_zero = line[line != 0]
            merged = []
            i = 0
            while i < len(non_zero):
                if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                    merged.append(2 * non_zero[i])
                    reward += 2 * non_zero[i]
                    i += 2
                else:
                    merged.append(non_zero[i])
                    i += 1
            new_board[row] = np.pad(merged, (0, 4 - len(merged)), 'constant')
        if not np.array_equal(new_board, self.board):
            self.board = new_board
            self.add_random_tile()
        return reward

    def move_right(self):
        self.board = np.fliplr(self.board)
        reward = self.move_left()
        self.board = np.fliplr(self.board)
        return reward

    def move_up(self):
        self.board = self.board.T
        reward = self.move_left()
        self.board = self.board.T
        return reward

    def move_down(self):
        self.board = self.board.T
        reward = self.move_right()
        self.board = self.board.T
        return reward

    def step(self, action):
        if action == 0:
            reward = self.move_left()
        elif action == 1:
            reward = self.move_right()
        elif action == 2:
            reward = self.move_up()
        elif action == 3:
            reward = self.move_down()
        done = not np.any(self.board == 0) and all([
            np.all(self.board[:, i] != self.board[:, i + 1]) for i in range(3)
        ]) and all([
            np.all(self.board[i, :] != self.board[i + 1, :]) for i in range(3)
        ])
        if 2048 in self.board:
            reward += 10000
            done = True
        state = self.board.flatten()
        max_tile = np.max(self.board)
        state = np.append(state, max_tile)
        return state, reward, done

    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.add_random_tile()
        self.add_random_tile()
        state = self.board.flatten()
        max_tile = np.max(self.board)
        state = np.append(state, max_tile)
        return state

# 深度 Q 网络类
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 经验回放缓冲区类
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

# 可视化函数
def visualize_board(board, ax):
    ax.clear()
    table = Table(ax, bbox=[0, 0, 1, 1])
    nrows, ncols = board.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # 定义颜色映射
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "yellow", "orange", "red"])

    for (i, j), val in np.ndenumerate(board):
        color = cmap(np.log2(val + 1) / np.log2(2048 + 1)) if val > 0 else "white"
        table.add_cell(i, j, width, height, text=val if val > 0 else "",
                       loc='center', facecolor=color)

    ax.add_table(table)
    ax.set_axis_off()
    plt.draw()
    plt.pause(0.1)

# 训练函数
def train():
    env = Game2048()
    input_size = 17
    output_size = 4
    model = DQN(input_size, output_size).to(device)
    target_model = DQN(input_size, output_size).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(capacity=10000)
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    update_target_freq = 10

    num_episodes = 1000
    fig, ax = plt.subplots()
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        done = False
        total_reward = 0
        while not done:
            visualize_board(env.board, ax)
            if random.random() < epsilon:
                action = random.randint(0, output_size - 1)
            else:
                q_values = model(state)
                action = torch.argmax(q_values, dim=1).item()

            next_state, reward, done = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            replay_buffer.add(state.cpu().squeeze(0).numpy(), action, reward, next_state.cpu().squeeze(0).numpy(), done)

            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)

                q_values = model(states)
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                next_q_values = target_model(next_states)
                next_q_values = next_q_values.max(1)[0]
                target_q_values = rewards + gamma * (1 - dones) * next_q_values

                loss = criterion(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            total_reward += reward

        if episode % update_target_freq == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {episode}: Total Reward = {total_reward}, Epsilon = {epsilon}")

    plt.close()

if __name__ == "__main__":
    train()