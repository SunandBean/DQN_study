# 신경망 구성
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_in, n_mid, n_out, Dueling):
        super(Net, self).__init__()
        self.Dueling = Dueling
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        # Dueling Network
        if self.Dueling == True:
            self.fc3_adv = nn.Linear(n_mid, n_out) # Advantage 함수 쪽 신경망
            self.fc3_v = nn.Linear(n_mid, 1) # 가치 V 쪽 신경망
        else:
            self.fc3 = nn.Linear(n_mid, n_out)
        
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        
        if self.Dueling == True:
            adv = self.fc3_adv(h2) # 이 출력은 ReLU를 거치지 않음
            val = self.fc3_v(h2).expand(-1, adv.size(1)) # 이 출력은 ReLU를 거치지 않음
            # val은 adv와 덧셈을 하기 위해 expand 메서드로 크기를 [minibatch*1]에서 [minibatch*2]로 변환
            # adv.size(1)은 2(출력할 행동의 가짓수)
            
            output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
            # val + adv 에서 adv의 평균을 뺀다
            # adv.mean(1, keepdim=True)로 열방향(행동의 종류 방향) 평균을 구함.
            # 크기는 [minibatch*1]이 됨
            # expand 메서드로 크기를 [minibatch*2]로 늘림
        else:
            output = self.fc3(h2)
        
        return output