"""
import numpy as np

a = np.array([x+1 for x in range(25)])
b = np.array([[0 for x in range(25)] for x in range(25)])
shift = -2
for g in range(25):
	b[g][(g+shift)%25]=1
# b[24][0]=1
print(np.sum(a*b,axis=1))
# print(b)
"""
import torch

# a = torch.LongTensor([x+1 for x in range(25))
batch = 50
a = torch.ones((25,batch)).long()
print(a)

k = [[0 for x in range(25)] for x in range(25)]
l =[0,6,12,18,24,3,9,10,16,22,1,7,13,19,20,4,5,11,17,23,2,8,14,15,21]
for g in range(25):
	k[g][l[g]]=1
# print(k)
# a[:]=100z
b = torch.LongTensor(k)
b = b.unsqueeze(2)
for x in range(3):
	for y in range(3):
		for d in range(3):
			
			c = torch.ones((25)).long()
			# print(b.shape)
			# print(a.shape)
			b = c.unsqueeze(x)
			b = c.unsqueeze(y)
			# print(b)
			try:
				s = torch.tensordot(a, b,dims = d)
				print(s)

			except:
				pass
	# 		break
	# 	break
	# break
# # print(b)
# shift = 1
# # for g in range(25):
# # 	b[g][(g+shift)%25]=1
# # b[24][0]=1
# # for x in range(3):
# # 	for y in range(3):
# # 		for d in range(3):
# x,y,d = 0,1,2
# C = torch.ones((50)).long()
# C = C.unsqueeze(x)
# C = C.unsqueeze(y)
# # print(C.shape)
# D = a*b
# # print(D.shape)

# E = torch.tensordot(D,C,dims=2)
# print(x,y,d,E.shape)
# print(E)
# # print(E[0])
# print(torch.sum((a*b),axis=1,out=a))
# # print(b)
# # def index(x,y):
# #     return (((x)%5)+5*((y)%5))

# # import numpy as np
# # A = np.zeros(25,dtype=np.int8)
# # A[:]=-1
# # temp = np.array([x for x in range(25)],dtype=np.uint64)
# # for x in range(5):
# # 	for y in range(5):
# # 		A[index(0*x+1*y, 2*x+3*y)] = temp[index(x, y)]
# # print(temp)
# # print(A)