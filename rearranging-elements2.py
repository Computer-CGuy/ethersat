import numpy as np

a = np.arange(25)
p = np.arange(25)

def index(x,y):
	return (((x)%5)+5*((y)%5))

for x in range(5):
	for y in range(5):
		a[index(0*x+1*y,2*x+3*y)] = p[index(x,y)]
	# 	a[index(x,y)] = p[index(x+2,y)]
	# a[x] = p[(x+1)%25]
print(a.tolist())
# indc = 25
# b = np.array([[0 for x in range(indc)] for x in range(indc)])

# # k = np.array([[26, 18, 4611686018427387907, 7784628224, 6308233216, 2130303778816, 246290604621824, 512, 792633534417207296, 35651584, 128, 3072, 26388279066624, 637534208, 20340965113856, 46179488366592, 844424930131968, 983040, 25165824, 14336, 3670016, 116, 2305843009213693955, 648518346341351424, 835584]])
# # k = np.arange(indc)

# # shift=2
# for g in range(indc):
# 	# b[g][(g+shift)%indc]=1
# 	b[g][(a[g])%indc]=1


# print(b.tolist())
# # print(np.sum(b*k,axis=1))
# # print(b)