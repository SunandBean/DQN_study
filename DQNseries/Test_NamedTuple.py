# 이 코드에서는 namedtuple을 사용한다.
# namedtuple을 사용하면 키-값의 쌍 형태로 값을 저장할 수 있다
# 그리고 키를 필드명으로 값에 접근할 수 있어 편리하다
# https://docs.python.jp/3/library/collections.html#collections.namedtuple
# 아래는 사용 예다

from collections import namedtuple

Tr = namedtuple('tr', ('name_a', 'value_b'))
Tr_object = Tr('이름A', 100)

print(Tr_object) # 출력 : tr(name_a='이름A', value_b = 100)
print(Tr_object.value_b) # 출력 : 100
print(Tr_object.name_a) # 출력 : 100