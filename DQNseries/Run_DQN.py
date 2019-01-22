# 구현에 사용할 패키지 임포트
from environment import Environment
from multiprocessing import Process, Queue

# 각 환경 설정
# Environment(Double, Dueling, PER)

def RunEnv(Double,Dueling,PER):
    Env = Environment(Double,Dueling,PER)
    Env.run()

DQN = Process(target=RunEnv, args=(False, False, False))

DDQN = Process(target=RunEnv, args=(True, False, False))
DuelingDQN = Process(target=RunEnv, args=(False, True, False))
PERDQN = Process(target=RunEnv, args=(False, False, True))

DuelingDDQN = Process(target=RunEnv, args=(True, True, False))
PERDDQN = Process(target=RunEnv, args=(True, False, True))
PERDuelingDQN = Process(target=RunEnv, args=(False, True, True))

PERDuelingDDQN = Process(target=RunEnv, args=(True, True, True))

# Start
DQN.start()

DDQN.start()
DuelingDQN.start()
PERDQN.start()

DuelingDDQN.start()
PERDDQN.start()
PERDuelingDQN.start()

PERDuelingDDQN.start()

# Finish
DQN.join()

DDQN.join()
DuelingDQN.join()
PERDQN.join()

DuelingDDQN.join()
PERDDQN.join()
PERDuelingDQN.join()

PERDuelingDDQN.join()