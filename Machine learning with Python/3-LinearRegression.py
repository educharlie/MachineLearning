import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

pageSpeed = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = 100 - (pageSpeed + np.random.normal(0, 0.1, 1000)) * 3

slope, intercept, r_value, p_value, std_error = stats.linregress(pageSpeed, purchaseAmount)
print r_value ** 2 #R-squared value shows a really good fit

fitLine = slope * pageSpeed + intercept

plt.scatter(pageSpeed, purchaseAmount)
plt.plot(pageSpeed, fitLine, c='r')
plt.show()
