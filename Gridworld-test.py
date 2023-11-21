from Gridworld import Gridworld
game = Gridworld(size=4, mode='static')
print(game.display())
# game.reward()
print(game.board.render_np())
