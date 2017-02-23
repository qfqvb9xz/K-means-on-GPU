import numpy as np
import random
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.gcf()
import matplotlib.cm as cm

# import PYCUDA modules and libraries
from pycuda import driver, compiler, gpuarray, tools
import sys
# the following module is used to mark the time stamps
import time

# -- initialize the device
import pycuda.autoinit

# additionally replace partial regenerate state by cuda k=5, n=100000, speedup=188
class KMeans_cu():
    def __init__(self, K, X, mu):
        self.K = K
        self.X = X
        self.N = len(X)
        self.mu = mu
        self.clusters = None
        self.assign = None
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

    def compute1(self):
        kernel_tmp = """
        __device__ float sqdist(float x1, float x2, float y1, float y2){
            return (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2);
        }

        __global__ void kmeans( float *data, const float * __restrict__ mu, unsigned int *assign, \
        float *newmu, unsigned int *num) {

            // each thread process 10*k data

            int id = blockIdx.x*blockDim.x + threadIdx.x;
            int lid = threadIdx.x;
            int K = %(K)s;
            int N = %(N)s;
            int size = %(size)s;
            __shared__ float dist[2 * %(K)s * %(size)s];
            __shared__ int number[%(K)s * %(size)s];

            for(int i=0; i<K; i++){                             // init shared memory
                dist[lid * 2 * K + i] = 0;
                dist[lid * 2 * K + i + K] = 0;
                number[lid * K + i] = 0;
            }

            __syncthreads();

            float point[2];
            while(id < N)
            {
                point[0] = data[2*id];
                point[1] = data[2*id + 1];
                float tmp = sqdist(point[0], mu[0], point[1], mu[1]);
                int key = 0;
                for(int i=1; i < K; i++){
                    if(sqdist(point[0], mu[2*i], point[1], mu[2*i+1])<tmp){
                        tmp = sqdist(point[0], mu[2*i], point[1], mu[2*i+1]);
                        key = i;        //key represents the closest cluster number
                    }
                }
                assign[id]=key;
                dist[lid * 2 * K + key*2] += point[0];
                dist[lid * 2 * K + key*2 + 1] += point[1];
                number[lid * K + key] += 1;

                id += size*(N/size/10/K + 1);
            }

            __syncthreads();        //all data recorded in shared memory


            for (unsigned int kn = 0; kn<K; kn++)
                for (unsigned int stride = size/2; stride > 0; stride /= 2)
                    if (lid < stride)
                    {
                        number[lid*K + kn] += number[(lid+stride)*K + kn];
                        dist[lid * 2 * K + kn*2] += dist[(lid+stride) * 2 * K + kn*2];
                        dist[lid * 2 * K + kn*2 + 1] += dist[(lid+stride) * 2 * K + kn*2 + 1];
                        __syncthreads();
                    }

            __syncthreads();        //reduction of shared data

            if (lid<K)
            {
                atomicAdd(&newmu[2*lid],dist[2*lid]);
                atomicAdd(&newmu[2*lid + 1],dist[2*lid + 1]);
                atomicAdd(&num[lid],number[lid]);
            }

        }

        """
        size = 128
        kernel = kernel_tmp % { 'N':self.N, 'K':self.K, 'size':size}

        mod = compiler.SourceModule(kernel)

        kmeans = mod.get_function("kmeans")
        data_gpu = gpuarray.to_gpu(self.X.astype(np.float32))
        mu_gpu = gpuarray.to_gpu(np.array(self.mu).astype(np.float32))
        assign_gpu = gpuarray.zeros((1, self.N), np.uint32)
        newmu_gpu = gpuarray.zeros((1, 2*self.K), np.float32)
        num_gpu = gpuarray.zeros((1, self.K), np.uint32)
        kmeans(data_gpu, mu_gpu, assign_gpu, newmu_gpu, num_gpu, block=(size, 1, 1), grid=(self.N/size/10/self.K + 1, 1, 1))
        assign = assign_gpu.get()
        newmu = newmu_gpu.get()     #np.array
        num = num_gpu.get()

        for i in range(self.K):
            if(num[0,i]!=0):
                newmu[0, i * 2] = newmu[0, i * 2] / num[0, i]
                newmu[0, i * 2 + 1] = newmu[0, i * 2 + 1] / num[0, i]

        self.mu = newmu.reshape(self.K,2).astype(np.float32).tolist()
        self.assign = assign

    def kernel1(self):
        self.oldmu = random.sample(data, k)
        loop = 1
        while not np.allclose(self.oldmu,self.mu):
            self.oldmu = self.mu
            self.compute1()
            print loop
            loop += 1

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

#data, k = _init_board_gauss(100000, 10)
#init_mu = random.sample(data, k)

import pickle
with open('data', 'rb') as fp:
    data = pickle.load(fp)
with open('k', 'rb') as fp:
    k = pickle.load(fp)
with open('init_mu', 'rb') as fp:
    init_mu = pickle.load(fp)

# init GPU

times = []
for average in xrange(5):
    kmeans_cu = KMeans_cu(k, X=data, mu=init_mu)
    start = time.time()
    kmeans_cu.kernel1()
    times.append(time.time() - start)
    kmeans_cu.plot_board()
ct = np.average(times)


print ct

