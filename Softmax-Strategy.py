import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt

n = 10
probs = np.random.rand(n)
record = np.zeros((n,2))

def softmax(av, tau=1.12):
    # 概率相加为1，tau越大，概率就相近不好区分
    softm = ( np.exp(av / tau) / np.sum( np.exp(av / tau) ) )
    return softm

def get_reward(prob, n=10):
    reward = 0;
    for i in range(n):
        if random.random() < prob:
            reward += 1
    return reward

def get_best_arm(record):
    arm_index = np.argmax(record[:,1],axis=0)
    return arm_index

def update_record(record,action,r):
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
    p = softmax(record[:,1],tau=0.7)
    # np.random.choice，np.arange(n)表示10个臂的序号列表，
    # p是根据平均奖励的概率分布，random.choice会从a中较多的选择对应p概率高的
    choice = np.random.choice(np.arange(n),p=p)
    r = get_reward(probs[choice])
    record = update_record(record,choice,r)
    mean_reward = ((i+1) * rewards[-1] + r)/(i+2)
    rewards.append(mean_reward)
ax.scatter(np.arange(len(rewards)),rewards)
plt.show()