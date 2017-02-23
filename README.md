# K-means-on-GPU
# instruction: put all files in same path, run data_init.py first, then op1~3 could work!

# input:	
* op1.py		optimization 1
* op2.py		optimization 2
* op3.py		optimization 3
* data_init.py	produce a dataset, ensure op1~3 have same data and initial mu, otherwise the rumtime would be nonsense.

# output:	cu_k-means_N100000_K5.png	a sample output file, you can try different sizes of data
	

# comment:
* 1. op2.py/op3.py can't work for very large data, e.g. 1000000 data points
* 2. output runtime could be stable after submit same file 2 or more times
* 3. some time the cluster would be unreasonable, duo to the k-means algorithm choose the initial mu randomly, this topic is beyound the project
* 4. cl.py for testing runtime of OpenCL
