import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--start',help='.',default=0,type=int)
parser.add_argument('--number',help='.',default=0,type=int)
args = parser.parse_args()

#number = 0
start = time.time()
for i in range(args.start,184903891,10000000):
    with open('train.csv','rb') as csvfile:
        section = pd.read_csv(csvfile,skiprows=range(1,i),nrows=i+10000000,header=0)
        data = section.loc[section['is_attributed']==1]
        data.to_csv(str(args.number)+'.csv',header=False,index=False)
        print i+10000000, data.shape[0]

end = time.time()
print 'downloads', len(downloads), 'time', start-end
print type(downloads), downloads.shape
 
