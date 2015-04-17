import sys
import numpy
from pyspark import SparkConf, SparkContext

f = int(sys.argv[1])
k = int(sys.argv[2])
T = int(sys.argv[3])
beta = float(sys.argv[4])
lam = float(sys.argv[5])
inputpath = sys.argv[6]
outputW = sys.argv[7]
outputH = sys.argv[8]
tao = 100

def loadmatrix(s):
	line = s.split(',')
	return [int(line[0]),int(line[1]),int(line[2])]

def makekey(s,param):
	x = s[0] * param.value[2] / param.value[0]
	if x >= param.value[2]:
		x = param.value[2]-1
	y = s[1] * param.value[2] / param.value[1]
	if y >= param.value[2]:
		y = param.value[2]-1
	main_key = (y - x + param.value[2]) % param.value[2]
	sub_key = x 
	return ([main_key,sub_key],s)

def sgd(iterator,w,h,lam,stepsize,N):
	n = 0
	decent_w = [] 
	decent_h = []
	hash_w = {}
	hash_h = {}
	mse = 0
	for record in iterator:
		i,j,v = record[1]
		hash_w[i] = 1
		hash_h[j] = 1
		decent_w = 2*lam*w[i]/N[0][i] - 2*(v - sum(w[i]*h[j]))*h[j]
		decent_h = 2*lam*h[j]/N[1][j] - 2*(v - sum(w[i]*h[j]))*w[i]
		w[i] = w[i] - stepsize*decent_w
		h[j] = h[j] - stepsize*decent_h
		mse += (v - sum(w[i]*h[j]))**2
		n += 1
	new_w = {}
	new_h = {}
	for key in hash_w:
		new_w[key] = w[key]  
	for key in hash_h:
		new_h[key] = h[key]  
	return [new_w,new_h,n,mse]

conf = SparkConf().setAppName('haha').setMaster('local')
sc = SparkContext(conf=conf)
data = inputpath
lines = sc.textFile(data)

m = lines.map(lambda s: loadmatrix(s))

# find the max user and max item

max_user,max_item = m.reduce(lambda a,b: (max(a[0],b[0]),max(a[1],b[1]))) 

# find the non-zero rating num

N_w = m.map(lambda s: (s[0],1)).countByKey().items()
N_h = m.map(lambda s: (s[1],1)).countByKey().items()
Nw_hash = {}
Nh_hash = {}
total_N = 0
for x in N_w:
	Nw_hash[x[0]] = x[1]
	total_N += x[1]
for x in N_h:
	Nh_hash[x[0]] = x[1]



broad_param = sc.broadcast([max_user,max_item,k])
broad_N = sc.broadcast([Nw_hash,Nh_hash])

key_m = m.map(lambda s: makekey(s,broad_param)) 
w = {}
h = {}

for i in range(1,max_user+1):
	w[i] = numpy.random.uniform(0,1,f)

for i in range(1,max_item+1):
	h[i] = numpy.random.uniform(0,1,f)

broad_lam = sc.broadcast(lam)
updatenum = 0

log_write = open('log.txt','w')
for t in range(T):
	stepsize = pow((tao+updatenum),-1*beta)
	broad_step = sc.broadcast(stepsize)
	mse = 0
	for i in range(k):
		broad_wh = sc.broadcast([w,h])
		
		stratum = sc.broadcast(i)
		temprdd = key_m.filter(lambda x: x[0][0] == stratum.value).map(lambda x: (x[0][1],x[1]))
		par_rdd = temprdd.partitionBy(k)
		haha = par_rdd.mapPartitions(lambda x:sgd(x,broad_wh.value[0],broad_wh.value[1],broad_lam.value,broad_step.value,broad_N.value))
		update_w,update_h,updatenum_delta,mse_delta = haha.glom().reduce(lambda a,b: [dict(a[0],**b[0]),dict(a[1],**b[1]),a[2]+b[2],a[3]+b[3]])
		updatenum += updatenum_delta
		mse += mse_delta
		for key in update_w:
			w[key] = update_w[key]
		for key in update_h:
			h[key] = update_h[key]
	mse /= total_N
	log_write.write(str(mse)+'\n')
log_write.close()

w_write = open(outputW,'w')	
h_write = open(outputH,'w')
for i in range(1,max_user+1):
	if w.has_key(i) == False:
		for j in range(f-1):
			w_write.write('0,')
		w_write.write('0\n')
	else:
		for j in range(f-1):
			w_write.write(str(w[i][j])+',')
		w_write.write(str(w[i][j])+'\n')

for i in range(f):
	for j in range(1,max_item):
		if h.has_key(j) == False:
			h_write.write('0,')
		else:
			h_write.write(str(h[j][i])+',')
	if h.has_key(max_item) == False:
		h_write.write('0\n')
	else:
		h_write.write(str(h[max_item][i])+'\n')

w_write.close()
h_write.close()





