# 구현에 사용할 패키지 임포트
from LunarLanderEnv import Environment
from multiprocessing import Process

# 각 환경 설정
# Environment(Double, Dueling, PER)

#Env = Environment(False,False,True)
#Env.run()

def RunEnv(Double,Dueling,PER):
    Env = Environment(Double,Dueling,PER)
    Env.run()

DQN = Process(target=RunEnv, args=(False, False, False))

PERDQN = Process(target=RunEnv, args=(False, False, True))

# Start
DQN.start()
PERDQN.start()


# Finish
DQN.join()
PERDQN.join()