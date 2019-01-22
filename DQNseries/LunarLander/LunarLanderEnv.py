# CartPole을 실행하는 환경 역할을 하는 클래스
import torch
import numpy as np
import gym
import matplotlib
import os
import datetime
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from LunarLanderAgent import Agent
from animation import display_frames_as_gif

# 상수 정의 
ENV = 'LunarLander-v2' # 태스크 이름

MAX_STEPS = 1000
NUM_EPISODES = 10000 # 최대 에피소드 수

class Environment:
    def __init__(self, Double, Dueling, PER):
        self.env = gym.make(ENV) # 태스크를 설정
        num_states = self.env.observation_space.shape[0] # 태스크의 상태 변수 수(8)를 받아옴
        num_actions = self.env.action_space.n # 태스크의 행동 가짓수(4)를 받아옴
        self.Double = Double
        self.Dueling = Dueling
        self.PER = PER
        self.agent = Agent(num_states, num_actions, Double, Dueling, PER) # 에이전트 역할을 할 객체를 생성

        self.EpiNum = []
        self.EpiScore = []
        
    def run(self): # 이 부분을 뜯어고쳐야함
        '''실행'''
        
        for episode in range(NUM_EPISODES): # 최대 에피소드 수만큼 반복
            done = 0
            score = 0
            observation = self.env.reset() # 환경 초기화
            
            state = observation # 관측을 변환 없이 그대로 상태 s로 사용
            state = torch.from_numpy(state).type(torch.FloatTensor) # NumPy 변수를 파이토치 Tensor로 변환
            state = torch.unsqueeze(state, 0) # size 8를 size 1*8로 변환
            
            #while not done:
            for step_num in range(MAX_STEPS):
                self.env.render() # 렌더링 켜기
                action = self.agent.get_action(state, episode) # 다음 행동을 결정
                
                # 행동 a_t를 실행해 다음 상태 s_{t+1}, reward와 done 플래그 값을 결정
                # action에 .item()을 호출해 행동 내용을 구함
                observation_next, reward, done, _ = self.env.step(action.item()) # reward와 info는 사용하지 않으므로 _로 처리
                if step_num >= 999:
                    reward -= step_num/10
                score += reward # 스코어 누적
                reward_mem = torch.FloatTensor([reward])
                # 보상을 부여하고 episode의 종료 판정 및 state_next 를 설정
                if done: # 에피소드 끝나면
                    state_next = None # 다음 상태가 없으므로 None으로
                else:
                    state_next = observation_next # 관측 결과를 그대로 상태로 사용
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor) # NumPy 변수를 파이토치 텐서로 변환
                    state_next = torch.unsqueeze(state_next, 0) # size 8를 size 1*8로 변환
                
                # 메모리에 경험을 저장
                self.agent.memorize(state, action, state_next, reward_mem)

                # TD 오차 메모리에 TD 오차를 저장
                # Prioritized Experience Replay 에서 추가됨
                if self.PER == True:
                    self.agent.memorize_td_error(0) # 여기서는 정확한 값 대신 0을 저장함
                
                # Experience Replay로 Q함수를 수정
                if self.PER == True:
                    self.agent.update_q_function(episode)
                else:
                    self.agent.update_q_function()
                
                # 관측 결과를 업데이트
                state = state_next
                
                # 에피소드 종료 처리
                if done:
                    self.EpiNum.append(episode)
                    self.EpiScore.append(score)
                    print('DQN with PER : %r'%(self.PER))
                    print('%d Episode: Finished in %d steps with %d score'%(episode, step_num, score))

                    # PER - TD 오차 메모리의 TD 오차를 업데이트 
                    if self.PER == True:
                        self.agent.update_td_error_memory()

                    # DDQN
                    if (episode % 2 == 0):
                        self.agent.update_target_q_function()
                    
                    break
                
        # 그림 그리기
        filename = "DQN_PER_%r_"%(self.PER) + datetime.datetime.now().strftime('%Y-%m-%d %H %M') + '.png'
        directory = './SaveResult'
        savepath = os.path.join(directory, filename)
        #plt.figure('%d%d%d'%(self.Double, self.Dueling, self.PER))
        plt.figure()
        plt.xlabel('num of episode')
        plt.ylabel('average steps')
        plt.title('DQN with PER : %r'%(self.PER))
        plt.grid()
        plt.savefig(savepath)
        plt.show()


                