import numpy as np
import matplotlib.pyplot as plt
from bspot import bidSPOT
from spot import biSPOT
from espot import ESPOT
from MOMspot import momSPOT
from drif_spot import DRSPOT
import pandas as pd
import time
f = './edf_stocks.csv'

P = pd.DataFrame.from_csv(f)

# stream
u_data = (P['DATE'] == '2017-02-09')
data = P['LOW'][u_data].values

# initial batch
u_init_data = (P['DATE'] == '2017-02-08') | (P['DATE'] == '2017-02-07') | (P['DATE'] == '2017-02-06')
init_data = P['LOW'][u_init_data].values


q = 1e-5             # risk parameter
d = 10                # depth
start = time.clock()
#s = ESPOT(q,d)     # bidSPOT object
#s = bidSPOT(q)  
s = biSPOT(q)  
#s =DRSPOT(q)
s.fit(init_data,data)     # data import
s.initialize()             # initialization step
results = s.run()     # run
end = time.clock()
t=end-start
print("Runtime is:",t) 
#del results['upper_thresholds'] # we can delete the upper thresholds
fig = s.plot(results)            # plot
