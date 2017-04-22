import numpy as np
import matplotlib.pyplot as plt

values = np.random.normal(10, 1, 10000)

plt.hist(values, 50)

#first moment
print ("First Moment: "),
print np.mean(values)
#second moment
print ("Second Moment: "),
print np.var(values)
#third moment
import scipy.stats as sp
print ("Third Moment: "),
print sp.skew(values)
#forth moment
print ("Forth Moment"),
print sp.kurtosis(values)

plt.show()
