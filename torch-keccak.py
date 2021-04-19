import torch

ROUNDS = 24
dtype = torch.long

def ROL64(a,offset):
	# offset = np.array([offset],dtype=dtype)
	# print((a // 2**(64-offset)).dtype)
	# print((a * 2**offset).dtype)
	return (a << offset) ^ (a >> (64-offset))

# def ROL64_1(a,offset):
# 	offset = np.array([offset],dtype=dtype)
# 	# print((a // 2**(64-offset)).dtype)
# 	# print((a * 2**offset).dtype)
# 	return (a * one[0]) ^ (a // 2**(one[1]))

def index(x,y):
	return (((x)%5)+5*((y)%5))

# KeccakRoundConstants = torch.LongTensor([
# 	0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
# 	0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
# 	0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
# 	0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
# 	0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
# 	0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
# 	0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
# 	0x8000000000008080, 0x0000000080000001, 0x8000000080008008])
KeccakRoundConstants = torch.zeros((25), dtype=dtype)
# ROL64_offsets = np.array([[0,64],[1,63],[2,62],[]])
one = torch.LongTensor([1])
KeccakRhoOffsets = torch.LongTensor([0 , 1 , 62, 28, 27, 36, 44, 6, 55, 20, 3, 10,43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14])



def theta(A,C,D):
	# C = np.zeros((5,batch),dtype=dtype)
	# D = np.zeros((5,batch),dtype=dtype)
	# print(C)
	for x in range(5):
		C[x] = 0
		for y in range(5):
			# print(index(x, y))
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
	# C = np.zeros((5,batch),dtype=dtype)
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


### CPU 
# NUMPY 100000 64358.35276040342  
# PYTORCH 100000 56451.490903139544
batch = 1000000
# state = np.array([0,0,0,0,1,0,0,0,9223372036854775808,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ],dtype=dtype)
state = torch.zeros([25,batch],dtype=dtype)
C = torch.zeros([25,batch],dtype=dtype)
D = torch.zeros([25,batch],dtype=dtype)
# state[:] = 0
import time
t0= time.time()
print(state.shape)
mine(state, C, D)
t1 = time.time() - t0
t1/=batch
print(1/t1)
# print(state[:,0])
# print(state[:,1])