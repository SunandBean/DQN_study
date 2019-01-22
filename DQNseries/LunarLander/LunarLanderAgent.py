# CartPole 태스크의 에이전트 클래스, 봉 달린 수레 자체라고 보면 된다
from LunarLanderBrain import Brain
class Agent:
    def __init__(self, num_states, num_actions, Double, Dueling, PER):
        ''' 태스크의 상태 및 행동의 가짓수를 설정 '''
        self.Double = Double
        self.Dueling = Dueling
        self.PER = PER
        self.brain = Brain(num_states, num_actions, Double, Dueling, PER) # 에이전트의 행동을 결정할 두뇌 역할 객체를 생성
        
    def update_q_function(self, episode = 0):
        ''' Q 함수를 수정 '''
        if self.PER == True:
            self.brain.replay(episode)
        else:
            self.brain.replay()
        
    def get_action(self, state, episode):
        ''' 행동을 결정 '''
        action = self.brain.decide_action(state, episode)
        return action
    
    def memorize(self, state, action, state_next, reward):
        ''' memory 객체에 state, action, state_next, reward 내용을 저장'''
        self.brain.memory.push(state, action, state_next, reward)

    # DDQN
    def update_target_q_function(self):
        ''' Traget Q-Network을 Main Q-Network와 맞춤 '''
        self.brain.update_target_q_network()

    # PER
    def memorize_td_error(self, td_error): # Prioritized Experience Replay 에서 추가됨
        ''' TD 오차 메모리에 TD 오차를 저장 '''
        self.brain.td_error_memory.push(td_error)
        
    def update_td_error_memory(self): # Prioritized Experience Replay 에서 추가됨
        ''' TD 오차 메모리의 TD 오차를 업데이트 '''
        self.brain.update_td_error_memory()
    