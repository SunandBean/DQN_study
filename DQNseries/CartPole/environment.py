# CartPole을 실행하는 환경 역할을 하는 클래스
import torch
import numpy as np
import gym
import matplotlib
import os
import datetime
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from agent import Agent
from animation import display_frames_as_gif

# 상수 정의 
ENV = 'CartPole-v0' # 태스크 이름
#ENV = 'MountainCar-v0'
#ENV = 'LunarLander-v2'
MAX_STEPS = 200 # 1 에피소드당 최대 단계 수
#MAX_STEPS = 300 # 1 에피소드당 최대 단계 수
NUM_EPISODES = 500 # 최대 에피소드 수

class Environment:
    def __init__(self, Double, Dueling, PER):
        self.env = gym.make(ENV) # 태스크를 설정
        num_states = self.env.observation_space.shape[0] # 태스크의 상태 변수 수(4)를 받아옴
        num_actions = self.env.action_space.n # 태스크의 행동 가짓수(2)를 받아옴
        self.Double = Double
        self.Dueling = Dueling
        self.PER = PER
        self.agent = Agent(num_states, num_actions, Double, Dueling, PER) # 에이전트 역할을 할 객체를 생성

        self.NumEpisode = []
        self.AvgSteps = []
        
    def run(self):
        '''실행'''
        episode_10_list = np.zeros(10) # 최근 10에피소드 동안 버틴 단계 수를 저장함 (평균 단계 수를 출력할 때 사용)
        complete_episodes = 0  # 현재까지 195단계를 버틴 에피소드 수
        episode_final = False # 마지막 에피소드 여부
        frames = [] # 애니메이션을 만들기 위해 마지막 에피소드의 프레임을 저장할 배열
        
        for episode in range(NUM_EPISODES): # 최대 에피소드 수만큼 반복
            observation = self.env.reset() # 환경 초기화
            
            state = observation # 관측을 변환 없이 그대로 상태 s로 사용
            state = torch.from_numpy(state).type(torch.FloatTensor) # NumPy 변수를 파이토치 Tensor로 변환
            state = torch.unsqueeze(state, 0) # size 4를 size 1*4로 변환
            
            for step in range(MAX_STEPS): # 1 에피소드에 해당하는 반복문
                #if episode_final is True: # 마지막 에피소드에서는 각 시각의 이미지를 frames에 저장한다.
                #    frames.append(self.env.render(mode='rgb_array'))
                    
                action = self.agent.get_action(state, episode) # 다음 행동을 결정
                
                # 행동 a_t를 실행해 다음 상태 s_{t+1}과 done 플래그 값을 결정
                # action에 .item()을 호출해 행동 내용을 구함
                observation_next, _, done, _ = self.env.step(action.item()) # reward와 info는 사용하지 않으므로 _로 처리
                
                # 보상을 부여하고 episode의 종료 판정 및 state_next 를 설정
                if done: # 단계 수가 200을 넘었거나 봉이 일정 각도 이상 기울면 done이 True가 됨
                    state_next = None # 다음 상태가 없으므로 None으로
                    
                    # 최근 10 에피소드에서 버틴 단계 수를 리스트에 저장
                    episode_10_list = np.hstack((episode_10_list[1:], step + 1))
                    
                    if step < 195:
                    #if step < 295:
                        reward = torch.FloatTensor([-1.0]) # 도중에 봉이 쓰러졌다면 페널티로 보상 -1을 부여
                        complete_episodes = 0 # 연속 성공 에피소드 기록을 초기화
                        
                    else:
                        reward = torch.FloatTensor([1.0]) # 봉이 서 있는 채로 에피소드를 마쳤다면 보상 1 부여
                        complete_episodes = complete_episodes + 1 # 연속 성공 에피소드 기록을 갱신

                    # 그림 그리기 위해 저장
                    self.NumEpisode.append(episode)
                    self.AvgSteps.append(episode_10_list[-1])

                else:
                    reward = torch.FloatTensor([0.0]) # 그 외의 경우는 보상 0을 부여
                    state_next = observation_next # 관측 결과를 그대로 상태로 사용
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor) # NumPy 변수를 파이토치 텐서로 변환
                    state_next = torch.unsqueeze(state_next, 0) # size 4를 size 1*4ㄹㄹ로 변환
                
                # 메모리에 경험을 저장
                self.agent.memorize(state, action, state_next, reward)

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
                    print('DQN with Double : %r, Dueling : %r, PER : %r'%(self.Double, self.Dueling, self.PER))
                    print('%d Episode: Finished after %d steps : 최근 10 에피소드의 평균 단계 수 = %.1lf'
                          %(episode, step + 1, episode_10_list.mean()))

                    # PER - TD 오차 메모리의 TD 오차를 업데이트 
                    if self.PER == True:
                        self.agent.update_td_error_memory()

                    # DDQN
                    if (episode % 2 == 0):
                        self.agent.update_target_q_function()
                    break
                    
                if episode_final is True:
                    # 애니메이션 생성 및 저장
                    #display_frames_as_gif(frames)
                    
                    break
                
                # 10 에피소드 연속으로 195단계를 버티면 태스크 성공
                if complete_episodes >= 5:
                    print('---- DQN with Double : %r, Dueling : %r, PER : %r ----'%(self.Double, self.Dueling, self.PER))
                    print('10 에피소드 연속 성공')
                    print('------------------------------------------------------')
                    # 그림 그리기
                    filename = "DQN_Double_%r_Dueling_%r_PER_%r_"%(self.Double, self.Dueling, self.PER) + datetime.datetime.now().strftime('%Y-%m-%d %H %M') + '.png'
                    directory = './SaveResult'
                    savepath = os.path.join(directory, filename)
                    plt.figure('%d%d%d'%(self.Double, self.Dueling, self.PER))
                    plt.scatter(self.NumEpisode, self.AvgSteps)
                    plt.xlabel('num of episode')
                    plt.ylabel('average steps')
                    plt.title('DQN with Double : %r, Dueling : %r, PER : %r'%(self.Double, self.Dueling, self.PER))
                    plt.grid()
                    plt.savefig(savepath)
                    plt.show()
                    episode_final = True # 다음 에피소드에서 애니메이션을 생성


                