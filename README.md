## Tutorial

### DQN-2048/Env2048Gui.py

[DQN 玩 2048 实战｜第一期！搭建游戏环境（附 PyGame 可视化源码）](https://www.bilibili.com/video/BV1bGQDYbEPp/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)

### DQN-2048/trainv1.py

[DQN 玩 2048 实战｜第二期！设计 ε 贪心策略神经网络，简单训练一下吧！](https://www.bilibili.com/video/BV1xWQaYMEEs/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)

## 库
`pip3 install numpy -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`

`pip3 install gym -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`

`pip3 install matplotlib -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`

`pip3 install tensorflow -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`

`pip3 install scipy -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`

`pip3 install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`

Gridwold.py模块依赖GridBoard.py


## 问题
>AttributeError: 'CartPoleEnv' object has no attribute 'seed'

`pip3 uninstall gym`

`pip3 install gym==0.25.2 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`

`pip3 install gym[box2d] -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`