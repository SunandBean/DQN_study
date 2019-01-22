# transition을 저장하기 위한 메모리 클래스
import random
import numpy as np

# 구현에 사용할 namedtuple 생성
from collections import namedtuple

Transition = namedtuple('Transition',('state','action','next_state','reward'))

class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY # 메모리의 최대 저장 건수
        self.memory = [] # 실제 transition을 저장할 변수
        self.index = 0 # 저장 위치를 가리킬 인덱스 변수
        
    def push(self, state, action, state_next, reward):
        ''' transition = (state, action, state_next, reward)을 메모리에 저장'''
        
        if len(self.memory) < self.capacity:
            self. memory.append(None) # 메모리가 가득차지 않은 경우
        
        # Transition 이라는 namedtuple을 사용해 키-값 쌍의 형태로 값을 저장
        self.memory[self.index] = Transition(state, action, state_next, reward)
        
        self.index = (self.index + 1) % self.capacity # 다음 저장할 위치를 한 자리 뒤로 수정
        
    def sample(self, batch_size):
        ''' batch_size 개수만큼 무작위로 저장된 transition을 추출 '''
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        ''' len 함수로 현재 저장된 transition 개수를 반환 '''
        return len(self.memory)

# TD 오차를 저장할 메모리 클래스
TD_ERROR_EPSILON = 0.0001 # 오차에 더해줄 바이어스

class TDerrorMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY # 메모리의 최대 저장 건수
        self.memory = [] # 실제 TD 오차를 저장할 변수
        self.index = 0 # 저장 위치를 가리킬 인덱스 변수
        
    def push(self, td_error):
        ''' TD 오차를 메모리에 저장 '''
        
        if len(self.memory) < self.capacity:
            self.memory.append(None) # 메모리가 가득 차지 않은 경우
            
        self.memory[self.index] = td_error
        self.index = (self.index + 1) % self.capacity # 다음 저장할 위치를 한 자리 뒤로 수정
        
    def __len__(self):
        ''' len 함수로 현재 저장된 개수를 반환 '''
        return len(self.memory)
    
    def get_prioritized_indexes(self, batch_size):
        ''' TD 오차에 따른 확률로 인덱스를 추출 '''
        
        # TD 오차의 합을 계산
        sum_absolute_td_error = np.sum(np.absolute(self.memory))
        sum_absolute_td_error += TD_ERROR_EPSILON * len(self.memory) # 충분히 작은 값을 더함
        
        # batch_size개 만큼 난수를 생성하고 오름차순으로 정렬
        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)
        
        # 위에서 만든 난수로 인덱스를 결정
        indexes = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                tmp_sum_absolute_td_error += (abs(self.memory[idx]) + TD_ERROR_EPSILON)
                
                idx += 1
                
            # TD_ERROR_EPSILON을 더한 영향으로 인덱스가 실제 개수를 초과했을 경우를 위한 보정
            if idx >= len(self.memory):
                idx = len(self.memory) - 1
            indexes.append(idx)
            
        return indexes
    
    def update_td_error(self, updated_td_errors):
        ''' TD 오차를 업데이트 '''
        self.memory = updated_td_errors