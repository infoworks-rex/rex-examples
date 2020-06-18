import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

register(
    id='FrozenLake-v3',
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4',
           'is_slippery':False}
)

env = gym.make('FrozenLake-v3')

# Q Table을 모두 0으로 초기화 한다. : 2차원 (number of state, action space) = (16,4)
Q = np.zeros([env.observation_space.n, env.action_space.n])

# discount 정의 => 미래의 reward를 현재의 reward 보다 조금 낮게 본다.
dis = 0.99

# 몇 번 시도를 할 것인가 (에피소드)
num_episodes = 2000

# 에피소드마다 총 리워드의 합을 저장하는 리스트
rList = []

for i in range(num_episodes) : 
    state = env.reset()
    
    rAll = 0
    done = False
    
    # Q learning 알고리즘
    while not done : 
        # Action 중에 가장 R(Reward)이 큰 Action을 고른다. 
        # 이 때, random noise 방식으로 decaying Exploit & Exploration 구현 
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i+1))
        
        # 해당 Action을 했을 때 environment가 변하고, 새로운 state, reward, done 여부를 반환 받음
        new_state, reward, done, _ = env.step(action)
        
        # Q = R + Q 
        Q[state, action] = reward + dis * np.max(Q[new_state, :])
        env.render()
        rAll += reward
        state = new_state
    
    print(i,"end with ",rAll)
    rList.append(rAll)
    
print("Success rate : "+str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)

plt.bar(range(len(rList)), rList, color="blue")
plt.show()