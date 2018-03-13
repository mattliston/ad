import numpy as np
import pandas as pd
import random
import argparse
import time
import re
import csv

start = time.time()

np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={'float': '{:12.8f}'.format, 'int': '{:4d}'.format})

def fixdate(x,pos): #convert date into interger
    for i in range(0,x.shape[0]):
        click_time = re.findall(r' .*',x[i,5])
        x[i,5] = int(click_time[0][1:3]+click_time[0][4:6]+click_time[0][7:9])
        if pos:
            attributed_time = re.findall(r' .*',x[i,6])
            x[i,6] = int(attributed_time[0][1:3]+attributed_time[0][4:6]+attributed_time[0][7:9])
    return x

pos = pd.read_csv('/home/mattliston/ad/pos.csv',header=0)
pos = pos.as_matrix()

print 'Starting...'
neg = np.zeros(pos.shape)
chunksize = 10000
n=0
count=1
for chunk in pd.read_csv('/home/mattliston/ad/neg.csv', chunksize=chunksize):
    chunk = chunk.as_matrix()
    if n == 456825:
        sample = random.sample(chunk,20)
        print type(sample),len(sample),neg.shape
        for i in sample:
            i[6] = np.random.randint(235959)
            i = np.reshape(i,(1,8))
            i = fixdate(i,False)

        neg[n:n+20] = sample
        break
          
    else:
        sample = random.sample(chunk,25)
        for i in sample:
            i[6] = np.random.randint(235959)
            i = np.reshape(i,(1,8))
            i = fixdate(i,False)
        neg[n:n+25]=sample

    print count*chunksize, n
    n+=25
    count+=1

neg = pd.DataFrame(neg)
neg.to_csv('neg_test_sample.csv',index=False,header=False)

#pos = fixdate(pos,True)
#pos = pd.DataFrame(pos)
#pos.to_csv('pos_sample.csv',index=False,header=False)

print 'pos',pos[-50:],'neg',neg[-50:]
print 'this script took', time.time()-start, 'seconds.'






