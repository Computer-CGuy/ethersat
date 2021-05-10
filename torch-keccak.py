import torch

ROUNDS = 24
dtype = torch.long

def ROL64(a,offset):
	return (a << offset) ^ (a >> (64-offset))


def index(x,y):
	return (((x)%5)+5*((y)%5))
KeccakRoundConstants = torch.zeros((25), dtype=dtype)

one = torch.LongTensor([1])[0]
KeccakRhoOffsets = torch.LongTensor([0 , 1 , 62, 28, 27, 36, 44, 6, 55, 20, 3, 10,43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14])

def theta(A,C,D):
	for x in range(5):
		C[x] = 0
		for y in range(5):
			C[x] ^= A[index(x, y)]
		D[x] = ROL64(C[x], 1)
	for x in range(5):
		for y in range(5):
			A[index(x, y)] ^= D[(x+1)%5] ^ C[(x+4)%5]

def rho(A,C,D):
	for x in range(5):
		for y in range(5):
			A[index(x, y)] = ROL64(A[index(x, y)], KeccakRhoOffsets[index(x, y)])

def pi(A,C,D):
	temp = torch.clone(A)
	for x in range(5):
		for y in range(5):
			A[index(0*x+1*y, 2*x+3*y)] = temp[index(x, y)]

def chi(A,C,D):
	for y in range(5):
		for x in range(5):
			C[x] = A[index(x, y)] ^ ((~A[index(x+1, y)]) & A[index(x+2, y)])
		for x in range(5):
			A[index(x, y)] = C[x]

def iota(A,indexRound):
	A[index(0, 0)] ^= KeccakRoundConstants[indexRound]

def mine(state,C,D):
	for i in range(ROUNDS):
		theta(state,C,D)
		rho(state,C,D)
		pi(state,C,D)
		chi(state,C,D)
		iota(state, i)
# 52.142,018.896071605
batch = 100000
state = torch.zeros([25,batch],dtype=dtype)
C = torch.zeros([25,batch],dtype=dtype)
D = torch.zeros([25,batch],dtype=dtype)


import time
t0= time.time()
print(state.shape)

mine(state, C, D)

t1 = time.time() - t0
t1/=batch
print(1/t1)