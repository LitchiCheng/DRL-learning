import numpy as np
import random
import pygame

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
        state = self.board.flatten()
        return state, reward, done

    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.add_random_tile()
        self.add_random_tile()
        return self.board.flatten()

# 颜色定义
COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46)
}

# 绘制游戏板
def draw_board(screen, board, tile_size, margin):
    for i in range(4):
        for j in range(4):
            value = board[i][j]
            color = COLORS.get(value, (0, 0, 0))
            pygame.draw.rect(screen, color,
                             (j * (tile_size + margin) + margin,
                              i * (tile_size + margin) + margin,
                              tile_size, tile_size))
            if value != 0:
                font = pygame.font.Font(None, 36)
                text = font.render(str(value), True, (0, 0, 0))
                text_rect = text.get_rect(center=(
                    j * (tile_size + margin) + margin + tile_size // 2,
                    i * (tile_size + margin) + margin + tile_size // 2
                ))
                screen.blit(text, text_rect)

# 主函数
def main():
    pygame.init()
    tile_size = 100
    margin = 10
    width = height = 4 * (tile_size + margin) + margin
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("2048 Game")

    game = Game2048()
    done = False
    clock = pygame.time.Clock()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    _, _, done = game.step(0)
                elif event.key == pygame.K_RIGHT:
                    _, _, done = game.step(1)
                elif event.key == pygame.K_UP:
                    _, _, done = game.step(2)
                elif event.key == pygame.K_DOWN:
                    _, _, done = game.step(3)

        screen.fill((187, 173, 160))
        draw_board(screen, game.board, tile_size, margin)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()