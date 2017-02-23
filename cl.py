import numpy as np
import pyopencl as cl
import pyopencl.array
import random
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.gcf()
import matplotlib.cm as cm
import sys
# the following module is used to mark the time stamps
import time


NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()
# Set up a command queue:
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx)


#-----------------------------------  host code  -----------------------------------
# additionally replace partial regenerate state by cuda k=5, n=100000, speedup=188
class KMeans_cu():
    def __init__(self, K, X, mu):
        self.K = K
        self.X = X
        self.N = len(X)
        self.mu = mu
        self.clusters = None
        self.assign = None

        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()
        # Set up a command queue:
        ctx = cl.Context(devs)
        queue = cl.CommandQueue(ctx)

        self.ctx = ctx
        self.queue = queue

    def plot_board(self):
        clusters = {}  # dictionary
        key = 0
        for x in self.X:
            bestmukey = self.assign[0, key]  # index to np.array!!
            try:
                clusters[bestmukey].append(x)
            except KeyError:
                clusters[bestmukey] = [x]
            key = key + 1
        self.clusters = clusters

        X = self.X
        fig = plt.figure(figsize=(5, 5))
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        if self.mu and self.clusters:
            mu = self.mu
            clus = self.clusters
            for m, clu in clus.items():
                cs = cm.spectral(1. * m / self.K)
                plt.plot(mu[m][0], mu[m][1], 'o', marker='*', markersize=12, color=cs)
                plt.plot(zip(*clus[m])[0], zip(*clus[m])[1], '.', markersize=8, color=cs, alpha=0.5)
        else:
            plt.plot(zip(*X)[0], zip(*X)[1], '.', alpha=0.5)

        tit = 'K-means with random initialization'
        pars = 'N=%s, K=%s' % (str(self.N), str(self.K))
        plt.title('\n'.join([pars, tit]), fontsize=16)
        plt.savefig('cu_k-means_N%s_K%s.png' % (str(self.N), str(self.K)), bbox_inches='tight', dpi=200)

    def compute(self):

        kernel_tmp = """
inline void AtomicAdd(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void kmeans(__global float *data, __global const float * __restrict__ mu, __global unsigned int *assign,
		     __global float *newmu, __global unsigned int *num, const unsigned int K, const unsigned int N, const unsigned int size,
		     __global int *barrier0, __global int *barrier1, __global float *oldmu, __global unsigned int *flag) {
	int gid = get_group_id(0)*get_local_size(0) + get_local_id(0);
	int lid = get_local_id(0);
	__local float dist[2 * %(K)s * %(size)s];
	__local int number[%(K)s * %(size)s];

	if(gid<K*2)
            oldmu[gid] = mu[gid];                           //outside of iterations
        while(flag[0]==1){
            flag[0] = 0;
            for(int i=0; i<K; i++){                             // init shared memory
                dist[lid * 2 * K + i] = 0;
                dist[lid * 2 * K + i + K] = 0;
                number[lid * K + i] = 0;
            }
            barrier0[get_group_id(0)] = 0;
            if (lid==0)
                barrier1[get_group_id(0)] = 1;
	    if (lid < N/size/10/K + 1)
		while(!barrier1[lid])
	    barrier(CLK_LOCAL_MEM_FENCE);

            float point[2];
            int id = gid;
            while(id < N)
            {
                point[0] = data[2*id];
                point[1] = data[2*id + 1];

		float tmp = (point[0]-oldmu[0])*(point[0]-oldmu[0])+(point[1]-oldmu[1])*(point[1]-oldmu[1]);
		int key = 0;
		for(int i=1; i < K; i++){
		    if((point[0]-oldmu[2*i])*(point[0]-oldmu[2*i])+(point[1]-oldmu[2*i + 1])*(point[1]-oldmu[2*i + 1]) <tmp){
			tmp = (point[0]-oldmu[2*i])*(point[0]-oldmu[2*i])+(point[1]-oldmu[2*i + 1])*(point[1]-oldmu[2*i + 1]);
			key = i;        //key represents the closest cluster number
                    }
		}
		assign[id]=key;
		dist[lid * 2 * K + key*2] += point[0];
                dist[lid * 2 * K + key*2 + 1] += point[1];
                number[lid * K + key] += 1;

		id += size*(N/size/10/K + 1);
            }
	    barrier(CLK_LOCAL_MEM_FENCE);

	    for (unsigned int kn = 0; kn<K; kn++){
                for (unsigned int stride = size/2; stride > 0; stride /= 2)
                    if (lid < stride)
                    {
                        number[lid*K + kn] += number[(lid+stride)*K + kn];
                        dist[lid * 2 * K + kn*2] += dist[(lid+stride) * 2 * K + kn*2];
                        dist[lid * 2 * K + kn*2 + 1] += dist[(lid+stride) * 2 * K + kn*2 + 1];
		        barrier(CLK_LOCAL_MEM_FENCE);
		    }

	    barrier(CLK_LOCAL_MEM_FENCE);

            if (lid<K)
            {
                AtomicAdd(&newmu[2*lid],dist[2*lid]);
                AtomicAdd(&newmu[2*lid + 1],dist[2*lid + 1]);
                atomic_add(&num[lid],number[lid]);
            }
            barrier1[get_group_id(0)] = 0;
            if (lid==0)
                barrier0[get_group_id(0)] = 1;
	    if (lid< N/size/10/K + 1)
		while(!barrier0[lid])

    	    barrier(CLK_LOCAL_MEM_FENCE);

	    if(gid<K){
		if(num[gid]!=0){
                        newmu[gid * 2] = newmu[gid * 2] / num[gid];
                        newmu[gid * 2 + 1] = newmu[gid * 2 + 1] / num[gid];
                    }
                    oldmu[gid * 2] = newmu[gid * 2];
                    oldmu[gid * 2+1] = newmu[gid * 2+1];

                    if(fabs(oldmu[gid * 2]-newmu[gid * 2])>1.e-6 || fabs(oldmu[gid * 2+1]-newmu[gid * 2+1])>1.e-6)
                        flag[0] = 1;
                }
            }

        }
}
"""
        size = 256
        flag = np.array([1]).astype(np.uint32)        


	grid = self.N/size/10/self.K + 1

	kernel = kernel_tmp %{'K': self.K, 'size': size}
        prg = cl.Program(self.ctx, kernel).build()
        
        data_gpu = cl.array.to_device(self.queue,self.X.astype(np.float32))
        mu_gpu = cl.array.to_device(self.queue,np.array(self.mu).astype(np.float32))
	flag_gpu = cl.array.to_device(self.queue, flag)
        assign_gpu = cl.array.zeros(self.queue, self.N, np.uint32)
        newmu_gpu = cl.array.zeros(self.queue, 2*self.K, np.uint32)
        num_gpu = cl.array.zeros(self.queue, self.K, np.uint32)
	assign = np.empty(self.N).astype(np.uint32)

        barrier0 = cl.array.zeros(self.queue, grid, np.uint32)
        barrier1 = cl.array.zeros(self.queue, grid, np.uint32)
        oldmu = cl.array.zeros(self.queue, 2*self.K, np.float32)
	queue = self.queue
        prg.kmeans(queue, (grid*size,), (size,), data_gpu.data, mu_gpu.data, assign_gpu.data, newmu_gpu.data, num_gpu.data, np.uint32(self.N), np.uint32(self.K), np.uint32(size), barrier0.data, barrier1.data, oldmu.data, flag_gpu.data)
#	cl.enqueue_copy(queue, assign, assign_gpu)
#        assign = assign_gpu.get()

#        newmu = newmu_gpu.get()     #np.array
#        num = num_gpu.get()


        #self.mu = newmu.reshape(self.K,2).astype(np.float32).tolist()
       # self.assign = assign

        #print num

# init data,K
def _init_board_gauss(N, k):
    n = float(N) / k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.01, 0.05)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) and abs(b) < 1:
                x.append([a, b])
        X.extend(x)
    X = np.array(X)[:N]
    return X,k

data, k = _init_board_gauss(1000000, 6)
init_mu = random.sample(data, k)

# init GPU
kmeans_cu = KMeans_cu(k, X=data, mu=init_mu)

start = time.time()
kmeans_cu.compute()
ct = time.time() - start
#kmeans_cu.plot_board()
print ct
