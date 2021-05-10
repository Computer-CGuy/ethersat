import jax.numpy as np
import jax


Batch = 1000


a = np.arange(64*Batch).reshape(64,Batch)
b = np.array([1,2,3,4,5]*(Batch//5)).astype(int)

# operand = np.array([False, False, False ,False, False, False,False])
# si = np.array([7, 8, 9, 10, 11 , 8,  7]).astype(int)
# jax.lax.slice(operand, si, b)
print("A")
print(a)
print("B")
print(b)
a[:,b]
# print(a[:,b].shape)