import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt

n = 10
probs = np.random.rand(n) #A
eps = 0.1

# 10 actions x 2 columns
# Columns: Count #, Avg Reward
# 两列，第一列是每个臂的次数，第二列是该臂的平均奖励
record = np.zeros((n,2))

# 奖励函数
def get_reward(prob, n=10):
    reward = 0;
    for i in range(n):
        if random.random() < prob:
            reward += 1
    return reward

# 计算最佳动作
def get_best_arm(record):
    # argmax函数功能，返回最大值的索引；
    # 若axis=1，表明按行比较，输出每行中最大值的索引，
    # 若axis=0，则输出每列中最大值的索引。
    arm_index = np.argmax(record[:,1],axis=0)
    return arm_index

# 更新记录
def update_record(record,action,r):
    # （次数*平均奖励 + 新奖励） / （次数+1），增量更新平均奖励
    new_r = (record[action,0] * record[action,1] + r) / (record[action,0] + 1)
    record[action,0] += 1
    record[action,1] = new_r
    return record

fig,ax = plt.subplots(1,1)
ax.set_xlabel("Plays")
ax.set_ylabel("Avg Reward")
fig.set_size_inches(9,5)
rewards = [0]
for i in range(500):
    if random.random() > eps:
        # 随机概率大于0.2，就从过去经验数据中取出平均奖励最高的动作
        choice = get_best_arm(record)
    else:
        # 否则随机选一个动作
        choice = np.random.randint(10)
    # 得到当前臂的奖励
    r = get_reward(probs[choice])
    # 更新奖励记录
    record = update_record(record,choice,r)
    # 计算平均奖励
    mean_reward = ((i+1) * rewards[-1] + r)/(i+2)
    rewards.append(mean_reward)
ax.scatter(np.arange(len(rewards)),rewards)
plt.show()