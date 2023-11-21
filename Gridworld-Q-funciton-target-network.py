import numpy as np
import torch
from Gridworld import Gridworld
# from IPython.display import clear_output
import random
from matplotlib import pylab as plt
import math
import copy

l1 = 64
l2 = 150
l3 = 100
l4 = 4

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3,l4)
)

model2 = copy.deepcopy(model) #A复制原始Q网络模型
model2.load_state_dict(model.state_dict()) #B复制原始Q网络的参数

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

gamma = 0.9
epsilon = 0.3

action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r',
}

from collections import deque
epochs = 5000
losses = []
mem_size = 1000 #A经验回放内存总大小
batch_size = 200 #B小排量随机自己的大小
replay = deque(maxlen=mem_size) #C经验回放的deque双端队列
max_moves = 50 #D如果单次超过50次还未找到目标就退出
h = 0
sync_freq = 500 #A设置更新频率
j=0
for i in range(epochs):
    game = Gridworld(size=4, mode='random')
    state1_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
    state1 = torch.from_numpy(state1_).float()
    status = 1
    mov = 0
    while(status == 1): 
        j+=1
        mov += 1
        qval = model(state1) #E
        qval_ = qval.data.numpy()
        if (random.random() < epsilon): #F
            action_ = np.random.randint(0,4)
        else:
            action_ = np.argmax(qval_)
        
        action = action_set[action_]
        game.makeMove(action)
        state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
        state2 = torch.from_numpy(state2_).float()
        reward = game.reward()
        done = True if reward > 0 else False
        exp =  (state1, action_, reward, state2, done) #G创建一条状态、动作、奖励、下一个状态的经验
        replay.append(exp) #H塞进队列
        state1 = state2
        
        if len(replay) > batch_size: #I当经验列表长度大于随机自己大小，开始小批量训练
            minibatch = random.sample(replay, batch_size) #J使用random.sample进行随机抽样子集
            state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch]) #K，将每一部分抽离成单独的张量，这里是状态张量
            action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch]) #这里是动作张量
            reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
            state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
            done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
            
            Q1 = model(state1_batch) #L重新计算小批量的Q值，得到梯度
            with torch.no_grad():
                Q2 = model(state2_batch) #M计算下一状态的Q值，不计算梯度
            
            Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2,dim=1)[0]) #N计算目标Q值
            X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze() #gather函数可以通过动作索引获取Q1张量的子集，这样只选择与实际被选择动作相关的Q值
            loss = loss_fn(X, Y.detach())
            print(i, loss.item())
            # clear_output(wait=True)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            if j % sync_freq == 0: #C当到达更新频率，将主模型的参数复制到目标网络
                model2.load_state_dict(model.state_dict())

        if reward != -1 or mov > max_moves: #O
            status = 0
            mov = 0
losses = np.array(losses)

plt.figure(figsize=(10,7))
plt.plot(losses)
plt.xlabel("Epochs",fontsize=22)
plt.ylabel("Loss",fontsize=22)

plt.show()