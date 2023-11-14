from Gridworld import Gridworld
game = Gridworld(size=4, mode='static')
game.display()
# game.reward()
print(game.board.render_np())