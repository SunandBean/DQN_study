# 에이전트의 두뇌 역할을 하는 클래스, DQN을 실제 수행한다.
# Q 함수를 딥러닝 신경망 형태로 정의

import random
import torch
from LunarLanderNet import Net
from torch import optim
import torch.nn.functional as F
import numpy as np
from LunarLanderRepMem import ReplayMemory, TDerrorMemory

# 구현에 사용할 namedtuple 생성

from collections import namedtuple

Transition = namedtuple('Transition',('state','action','next_state','reward'))

# 상수 정의
GAMMA = 0.99 # 시간 할인율
BATCH_SIZE = 32
CAPACITY = 10000
#random.seed(1)
    
# 기존의 DQN의 Brain과 합치는 방법에 대해서 좀 더 고민해보기 
class Brain:
    def __init__(self, num_states, num_actions, Double, Dueling, PER):
        self.num_actions = num_actions # 행동 가짓수(2)를 구함
        self.Double = Double
        self.Dueling = Dueling
        self.PER = PER
        
        # transition을 기억하기 위한 메모리 객체 생성
        self.memory = ReplayMemory(CAPACITY)
        
        # 신경망 구성
        n_in, n_mid, n_out = num_states, 512, num_actions
        self.main_q_network = Net(n_in, n_mid, n_out, Dueling) # Net 클래스를 사용
        self.target_q_network = Net(n_in, n_mid, n_out, Dueling) # Net 클래스를 사용
        print(self.main_q_network) # 신경망의 구조를 출력
        
        # 최적화 기법을 선택
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.0001)
        
        # PER - TD 오차를 기억하기 위한 메모리 객체 생성
        if self.PER == True:
            self.td_error_memory = TDerrorMemory(CAPACITY)

    def replay(self, episode = 0):
        ''' Experience Replay로 신경망의 결합 가중치 학습 '''
        
        # 1. 저장된 transition 수 확인
        if len(self.memory) < BATCH_SIZE:
            return
        
        # 2. 미니배치 생성
        if self.PER == True:
            self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch(episode)
        else:
            self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()
        
        # 3. 정답신호로 사용할 Q(s_t, a_t)를 계산
        self.expected_state_action_values = self.get_expected_state_action_values()
        
        # 4. 결합 가중치 수정
        self.update_main_q_network()
        
    def decide_action(self, state, episode):
        '''현재 상태로부터 행동을 결정함'''
        # e-greedy 알고리즘에서 서서히 최적행동의 비중을 늘린다
        epsilon = 0.5 * (1 / (episode +1))
        
        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval() # 신경망을 추론 모드로 전환
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)
            # 신경망 출력의 최댓값에 대한 인덱스 = max(1)[1]
            # .view(1,1)은 [torch.LongTensor of size 1]을 size 1*1로 변환하는 역할을 함
            
        else:
            # 행동을 무작위로 반환 (0 혹은 1)
            action = torch.LongTensor([[random.randrange(self.num_actions)]]) #행동을 무작위로 반환(0 혹은 1)
            # action은 [torch.LongTensor of size 1*1] 형태가 된다.
            
        return action
    
    def make_minibatch(self, episode = 0):
        '''2. 미니배치 생성'''
        

        if self.PER == True:
            # 2.1 PER - 메모리 객체에서 미니배치를 추출
            # def make_minibatch(self, episode):
            if episode < 30:
                transitions = self.memory.sample(BATCH_SIZE)
            else:
                # TD 오차를 이용해 미니배치를 추출하도록 수정
                indexes = self.td_error_memory.get_prioritized_indexes(BATCH_SIZE)
                transitions = [self.memory.memory[n] for n in indexes]
        else:
            # 2.1 메모리 객체에서 미니배치를 추출
            transitions = self.memory.sample(BATCH_SIZE)

        # 2.2 각 변수를 미니배치에 맞는 형태로 변형
        # transitions는 각 단계별로 (state, action, state_next, reward) 형태로 BATCH_SIZE 개수만큼 저장됨
        # 다시 말해, (state, action, state_next, reward) * BATCH_SIZE 형태가 된다.
        # 이를 미니배치로 만들기 위해
        # (state*BATCH_SIZE, action*BATCH_SIZE), state_next*BATCH_SIZE, reward*BATCH_SIZE)
        # 형태로 변환한다.
        
        batch = Transition(*zip(*transitions))
        
        # 2.3 각 변수의 요소를 미니배치에 맞게 변형하고, 신경망으로 다룰 수 있게 Variable로 만든다
        # state를 예로 들면, [torch.FloatTensor of size 1*4] 형태의 요소가 BATCH_SIZE 개수만큼 있는 형태다
        # 이를 torch.FloatTensor of size BATCH_SIZE*4 형태로 변형한다
        # 상태, 행동, 보상, non_final 상태로 된 미니배치를 나타내는 Variable을 생성
        # cat은 Concatenates(연접)를 의미한다.
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        return batch, state_batch, action_batch, reward_batch, non_final_next_states
    
    def get_expected_state_action_values(self):
        ''' 정답 신호로 사용할 Q(s_t,a_t)를 계산'''
        
        # 3.1 신경망을 추론 모드로 전환
        self.main_q_network.eval()
        self.target_q_network.eval()
        
        # 3.2 신경망으로 Q(s_t, a_t)를 계산
        # self.model(state_batch)은 왼쪽, 오른쪽에 대한 Q값을 출력하며
        # [torch.FloatTensor of size BATCH_SIZEx2] 형태다
        # 여기서부터는 실행한 행동 a_t에 대한 Q값을 계산하므로 action_batch에서 취한 행동
        # a_t가 왼쪽이냐 오른쪽이냐에 대한 인덱스를 구하고, 이에 대한 Q값을 gather메서드로 모아온다.
        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)
        
        # 3.3 max{Q(s_t+1, a)}값을 계산한다. 이때 다음 상태가 존재하는지에 주의해야 한다
        
        # cartpole이 done 상태가 아니고, next_state가 존재하는지 확인하는 인덱스 마스크를 만듬
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, self.batch.next_state)))
        
        # 먼저 전체를 0으로 초기화
        next_state_values = torch.zeros(BATCH_SIZE)

        # Double DQN
        if self.Double == True:        
            a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)
        
            # 다음 상태에서 Q값이 최대가 되는 행동 a_m을 Main Q-Network로 계산
            # 마지막에 붙은 [1]로 행동에 해당하는 인덱스를 구함
            a_m[non_final_mask] = self.main_q_network(self.non_final_next_states).detach().max(1)[1]
            
            # 다음 상태가 있는 것만을 걸러내고, size 32를 32*1로 변환
            a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)
            
            # 다음 상태가 있는 인덱스에 대해 행동 a_m의 Q값을 target Q-Network로 계산
            # detach() 메서드로 값을 꺼내옴
            # squeeze()메서드로 size[minibatch*1]을 [minibatch]로 변환
            next_state_values[non_final_mask] = self.target_q_network(
                self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()
        else:
            # 다음 상태가 있는 인덱스에 대한 최대 Q값을 구한다
            # 출력에 접근해서 열방향 최댓값(max(1))이 되는 [값, 인덱스]를 구한다
            # 그리고 이 Q값(인덱스 = 0)을 출력한 다음
            # detach 메서드로 이 값을 꺼내온다
            next_state_values[non_final_mask] = self.target_q_network(self.non_final_next_states).max(1)[0].detach()
        
        # 3.4 정답신호로 사용할 Q(s_t, a_t) 값을 Q러닝 식으로 계산
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values
        
        return expected_state_action_values
    
    def update_main_q_network(self):
        ''' 4. 결합 가중치 수정 '''
        
        # 4.1 신경망을 학습 모드로 전환
        self.main_q_network.train()
        
        # 4.2 손실함수를 계산(smooth_l1_loss는 Huber 함수)
        # expected_state_action_values은 size가 [minibatch]이므로 unsqueeze해서 [minibatch*1]로 만듦
        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))
        
        # 4.3 결합 가중치를 수정
        self.optimizer.zero_grad() # 경사를 초기화
        loss.backward() # 역전파 계산
        self.optimizer.step() # 결합 가중치 수정
        
    def update_target_q_network(self): # DDQN에서 추가됨
        ''' Target Q-Network을 Main Q-Network와 맞춤 '''
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    def update_td_error_memory(self): # Prioritized Experience Replay 에서 추가됨
        ''' TD 오차 메모리에 저장된 TD 오차를 업데이트 '''
        
        # 신경망을 추론 모드로 전환
        self.main_q_network.eval()
        self.target_q_network.eval()
        
        # 전체 transition으로 미니배치를 생성
        transitions = self.memory.memory
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        # 신경망의 출력 Q(s_t, a_t)를 계산
        state_action_values = self.main_q_network(state_batch).gather(1, action_batch)
        
        # cartpole이 done 상태가 아니고, next_state가 존재하는지 확인하는 인덱스 마스크를 만듦
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        
        # 먼저 전체를 0으로 초기화, 크기는 기억한 transition 개수만큼
        next_state_values = torch.zeros(len(self.memory))
        a_m = torch.zeros(len(self.memory)).type(torch.LongTensor)
        
        # 다음 상태에서 Q값이 최대가 되는 행동 a_m을 Main Q-Network로 계산
        # 마지막에 붙은 [1]로 행동에 해당하는 인덱스를 구함
        a_m[non_final_mask] = self.main_q_network(non_final_next_states).detach().max(1)[1]
        
        # 다음 상태가 있는 것만을 걸러내고, size 32를 32*1 로 변환
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)
        
        # 다음 상태가 있는 인덱스에 대해 행동 a_m의 Q값을 target Q-Network로 계산
        # detach() 메서드로 값을 꺼내옴
        # squeeze() 메서드로 size[minibatch*1]을 [minibatch]로 변환
        next_state_values[non_final_mask] = self.target_q_network(
            non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()
        
        # TD 오차를 계산
        td_errors = (reward_batch + GAMMA * next_state_values) - state_action_values.squeeze()
        # state_action_values는 size[minibatch*1]이므로 squeeze 메서드로 size[minibatch]로 변환
        
        # TD 오차 메모리를 업데이트. Tensor를 detach() 메서드로 꺼내와 NumPy 변수로 변환하고
        # 다시 파이썬 리스트로 반환
        self.td_error_memory.memory = td_errors.detach().numpy().tolist()
    