import numpy as np

ROUNDS = 24

def ROL64(a,offset):
	offset = np.array([offset],dtype=np.uint64)
	# print((a // 2**(64-offset)).dtype)
	# print((a * 2**offset).dtype)
	return (a << offset) ^ (a >> (64-offset))

# def ROL64_1(a,offset):
# 	offset = np.array([offset],dtype=np.uint64)
# 	# print((a // 2**(64-offset)).dtype)
# 	# print((a * 2**offset).dtype)
# 	return (a * one[0]) ^ (a // 2**(one[1]))

def index(x,y):
	return (((x)%5)+5*((y)%5))

KeccakRoundConstants = np.array([
	0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
	0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
	0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
	0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
	0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
	0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
	0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
	0x8000000000008080, 0x0000000080000001, 0x8000000080008008],dtype=np.uint64)

# ROL64_offsets = np.array([[0,64],[1,63],[2,62],[]])
one = np.array([1,63],dtype=np.uint64)
one = 2**one

KeccakRhoOffsets = np.array([0 , 1 , 62, 28, 27, 36, 44, 6, 55, 20, 3, 10,
	43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14],dtype=np.uint64)

def theta(A):
	C = np.zeros((5,batch),dtype=np.uint64)
	D = np.zeros((5,batch),dtype=np.uint64)
	# print(C)
	for x in range(5):
		C[x] = 0
		for y in range(5):
			# print(index(x, y))
			C[x] ^= A[index(x, y)]
	D = ROL64(C, 1)
	for x in range(5):
		for y in range(5):
			A[index(x, y)] ^= D[(x+1)%5] ^ C[(x+4)%5]

def rho(A):
	for x in range(5):
		for y in range(5):
			A[index(x, y)] = ROL64(A[index(x, y)], KeccakRhoOffsets[index(x, y)])

def pi(A):
	temp = np.copy(A)
	for x in range(5):
		for y in range(5):
			A[index(0*x+1*y, 2*x+3*y)] = temp[index(x, y)]

def chi(A):
	C = np.zeros((5,batch),dtype=np.uint64)
	for y in range(5):
		for x in range(5):
			C[x] = A[index(x, y)] ^ ((~A[index(x+1, y)]) & A[index(x+2, y)])
		for x in range(5):
			A[index(x, y)] = C[x]

def iota(A,indexRound):
	A[index(0, 0)] ^= KeccakRoundConstants[indexRound]



batch = 1
state = np.array([x for x in range(25)],dtype=np.uint64)
# state = np.zeros([25,1],dtype=np.uint64)
# state[:] = 0
import time
t0= time.time()

print(state.shape)
for i in range(ROUNDS):
	# print(i)
	theta(state)
	rho(state)
	pi(state)
	chi(state)
	iota(state, i)
print(state)
t1 = time.time() - t0
t1/=batch
print(1/t1)
# print(state[:,0])
# print(state[:,1])