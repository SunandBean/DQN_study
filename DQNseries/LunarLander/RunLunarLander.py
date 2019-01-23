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

DDQN = Process(target=RunEnv, args=(True, False, False))

PERDDQN = Process(target=RunEnv, args=(True, False, True))

# Start
DDQN.start()
PERDDQN.start()

# Finish
DDQN.join()
PERDDQN.join()