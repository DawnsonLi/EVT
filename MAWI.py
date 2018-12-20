import numpy as np
import matplotlib.pyplot as plt
#from bspot import bidSPOT
from spot import biSPOT
from drif_spot import DRSPOT
from middle_spot import MISPOT
from MOMspot import momSPOT
import pandas as pd
import time
#no label
f17 = './mawi_170812_50_50.csv'
f18 = './mawi_180812_50_50.csv'

P17 = pd.DataFrame.from_csv(f17)
P18 = pd.DataFrame.from_csv(f18)

X17 = P17['rSYN'].values
X18 = P18['rSYN'].values

n_init = 1000
init_data = X17[-n_init:]     # initial batch
data = X18                # stream

q = 1e-4             # risk parameter

start = time.clock()

#s = momSPOT(q)
#s = biSPOT(q)         # SPOT object
#s = DRSPOT(q)
s = MISPOT(q)
s.fit(init_data,data)     # data import
s.initialize()         # initialization step
results = s.run()     # run

end = time.clock()
t=end-start
print("Runtime is:",t) 

s.plot(results)     # plot