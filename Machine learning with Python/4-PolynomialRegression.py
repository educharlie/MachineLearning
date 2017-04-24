import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

np.random.seed(2)
pageSpeed = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50, 10, 1000) / pageSpeed

x = np.array(pageSpeed)
y = np.array(purchaseAmount)
p4 = np.poly1d(np.polyfit(x, y, 3))

xp = np.linspace(0, 7, 100)

r2 = r2_score(y, p4(x))

print r2 #r-square

plt.scatter(x, y)
plt.plot(xp, p4(xp), c='r')
plt.show()
