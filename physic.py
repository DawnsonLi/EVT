#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from espot import ESPOT
from MOMspot import momSPOT
from spot import biSPOT
from bspot import bidSPOT
f = './physics.dat'
r = open(f,'r').read().split(',')
X = np.array(list(map(float,r)))
import time
n_init = 2000
init_data = X[:n_init]     # initial batch
data = X[n_init:]          # stream

q = 1e-3                 # risk parameter
d = 450                  # depth parameter
start = time.clock()
s = biSPOT(q)
#s = ESPOT(q,d)         # biDSPOT object
#s = momSPOT(q)
#s = bidSPOT(q,d)
s.fit(init_data,data)     # data import
s.initialize()               # initialization step
results = s.run()        # run
end = time.clock()
t=end-start
print("Runtime is:",t) 
s.plot(results)          # plot