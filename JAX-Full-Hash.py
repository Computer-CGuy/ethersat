import jax.numpy as np
from jax import jit
from jax import lax

dtype = np.uint32

FNV_PRIME = 0x01000193
print(FNV_PRIME)
@jit
def fnv(v1,v2):
	# print("fnved")
	return ((v1 * FNV_PRIME) ^ v2)# % 2**32   

def timed(runer,rate=False,batch=1):
    global times
    import time
    global mix,seed_init,dataset,mix_hash
    t0 = time.time()
    ret = (runer(mix,seed_init,dataset,mix_hash))[0][0]
    # print(ret)
    ret.to_py()
    ret= None
    t1 = time.time()
    if(rate):
        print(f"{batch*times/((t1-t0)):,}")
        return ret
    print(f"{(t1-t0):,}s")
    return ret

@jit
def fit(mix1):
	# print(i)
	# print("Called")
	global seed_init,dataset,mix_hash
	# for i in range(64):
		# p = fnv(i^seed_init, mix[i%32])
		# newdata = dataset[:,p]
		# # print(newdata.shape)
		# # print("A"*200)
		# # print(mix.shape)
		# # print(newdata.shape)
		# mix = fnv(mix, newdata)
	# mix = lax.reduce(operands, init_values, computation, dimensions)
	# mix1 = lax.fori_loop(0, 64, 
	# 		lambda i, mix: fnv(mix, dataset[:,fnv(i^seed_init, mix[i%32])]),mix1)
	# fnv(i^seed_init, mix[i%32])
	
	# mix1 = lax.fori_loop(0, 64, 
	# 		lambda i, mix: fnv(mix, dataset[:,[1]*(batch)]),mix1)
	for x in range(64):
		mix1 = fnv(mix1,2)
	# mix1 = lax.fori_loop(0, 64, 
	# 		lambda i, mix: mix*2,mix1)
		# mix = fnv(mix, dataset[:,fnv(i^seed_init, mix[i%32])])		
	# cmax = []
	# for i in range(0,16,4):
	# 	cmax.append(fnv(fnv(fnv(mix[i], mix[i+1]), mix[i+2]), mix[i+3]))
	# 	mix_hash =
	# ind-=1
	# re
	return mix1

@jit
def loop(mix,seed_init,dataset,mix_hash):
	# print(times)
	return lax.fori_loop(0, times, 
						lambda i,mix: fit(mix), mix)
	# lax.while_loop(ind<0,fit,(ind,mix,seed_init,dataset,mix_hash))
# mix is 64*batch

batch = 20000
batch = 200
batch = 10000

mix = np.arange(64*batch,dtype=dtype).reshape(64,batch)
mix_hash = np.arange(32*batch,dtype=dtype).reshape(32,batch)
seed_init = np.array([0x1230],dtype=dtype)
dataset = np.arange((1000*64),dtype=dtype).reshape(64,1000)

times = 214789
times = 20
# times = 2000
        # 9223372036854775807
        # 2147483647
from jax import make_jaxpr
# print(loop(mix, seed_init, dataset, mix_hash))
# print(make_jaxpr(loop)(mix,seed_init,dataset,mix_hash))
print(f"{times*batch:,}")
print(timed(loop,rate=True,batch=batch))
# x = np.ones((2,3))
# x = fit(mix,seed_init,dataset,mix_hash)
# print()