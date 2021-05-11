import numpy as np

from jax.lib import xla_client as xc
xops = xc.ops


index_limit = 25


# Constants

Batch = 10000

times = 1
def timed(runer,args,rate=False,batch=1):
    global times
    import time
    ret = runer(args)
    # ret = runer(args)
    t0 = time.time()
    a = ret[0].to_py()
    t1 = time.time()
    # print(a)
    if(rate):
        print(f"{batch*times/((t1-t0)):,}")
        return ret
    print(f"{(t1-t0):,}s")
    return ret
def run(c):
	pass

# param_shape = xc.Shape.array_shape(np.dtype(np.float32), (max_index,Batch))
	
def timed2(runer,args,rate=False,batch=1):
    global times
    import time
    t0 = time.time()
    print(runer[args])
    ret= None
    t1 = time.time()
    if(rate):
        print(f"{batch*times/((t1-t0)):,}")
        return ret
    print(f"{(t1-t0):,}s")
    return ret
    
def run(c,batch=1,build=None):
    use_tpu = False
    if(use_tpu):
        cpu_backend = xb._get_tpu_driver_backend("tpu")
    else:
        cpu_backend = xc.get_local_backend("cpu")
    if(not build):
        computation = c.Build()
    else:
        computation = build
    
    compiled_computation = cpu_backend.compile(computation)

    As = np.random.randint(0,0xFFFFFF-1,(64+1,batch),dtype=dtype)
    
    # state = [cpu_backend.buffer_from_pyval(dtype(0)),cpu_backend.buffer_from_pyval(dtype(16))]
    state = []
    print("Evalving")
    for x in range(64+1):
        state.append(cpu_backend.buffer_from_pyval(As[x]))
    print("Started")
    out = timed(compiled_computation.execute,state,rate=True,batch=batch)
    # print((out[0].to_py()))
    
def generator():
    mix_shape = [xc.Shape.array_shape(np.dtype(dtype),(batch,)) for x in range(64)]
    # newdata = [xc.Shape.array_shape(np.dtype(dtype),(index_limit,64,batch,))]
    seed_init = [xc.Shape.array_shape(np.dtype(dtype),(batch,)) for x in range(1)]
    return [mix_shape]+[seed_init]

def paramaterize(c,tups):
    ret = []
    cnt = 0
    # print(tups)
    for x in range(len(tups)):
        inps = []
        # print(tups[x])
        for y in range(len(tups[x])):
            inps.append(xops.Parameter(c, cnt, tups[x][y]))
            cnt+=1
        ret.append(inps)
    return ret


def fnv(v1, v2):
    return xops.Xor(xops.Mul(v1,FNV_PRIME),v2)

def fnv1(v1, x2):
    ret = []

    for x in range(len(v1)):
        x1 =v1[x]
        # x2 = v2[x]
        # x2 = xops.Slice(v2,[x],[1],[0])
        ret.append(xops.Xor(xops.Mul(x1,FNV_PRIME),x2))
    return ret
    # return xops.Xor(xops.Mul(v1,FNV_PRIME),v2)

# def middle():
dtype = np.int32

miner = xc.XlaBuilder("W Loop")
FNV_PRIME = xops.Constant(miner, dtype(0x01000193))
converted = {}
batch = 10000

num = xc.Shape.array_shape(np.dtype(dtype),(batch,))
tup = generator()
# in_tuple_shape = xc.Shape.tuple_shape(in_tuple_shape)
# tup = xops.dataset(miner,0,in_tuple_shape)
mix,seed_init = paramaterize(miner, tup)



# def getElement(py_tuple,i):
#     param_shape = xc.Shape.array_shape(np.dtype(dtype), (Batch,))
#     tcb = xc.XlaBuilder("getElement")
#     tup = xops.Parameter(tcb, 0, param_shape)

#     test_computation = tcb.Build()
#     # print(test_computation)
#     return xops.Conditional(dtype(i), [test_computation for x in range(len(py_tuple))], py_tuple)

ones = xops.Constant(miner,dtype(1))
zeros = xops.Constant(miner,dtype(1))
twos = xops.Constant(miner,dtype(2))
def getArrayElement(arr,i):
    return xops.DynamicSlice(arr[0],[i],[1])


for x in range(64):
    i = xops.Constant(miner, dtype(x))
    p = fnv(xops.Xor(i,seed_init[0]), mix[x])
    # newdata = getArrayElement(dataset,p)
    # print(mix,newdata)
    mix = fnv1(mix, twos)
    # mix = fnv1(mix,newdata)
    # TODO: Implement Map Function
ret = []
for x in range(0,32,4):
    ret.append(fnv(mix[x], fnv(mix[x+1],fnv(mix[x+2],mix[x+3]))))
    # ret.append(fnv(getElement(mix,x),fnv(getElement(mix,x+1), fnv(getElement(mix,x+2), getElement(mix,x+3)))))

output = xops.Tuple(miner,ret)

# ans = SHA256_hash(w,rolls,consts)
# ans = [counter_,rounds_]+ans

# output = xops.Tuple(miner,ans)

# body_computation = miner.build()

run(miner,batch=batch)