# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

np.random.seed(2)
pageSpeed = np.random.normal(3.0, 1.0, 100)
purchaseAmount = np.random.normal(50.0, 30.0, 100) / pageSpeed

#take 80% for training
trainX = pageSpeed[:80]
trainY = purchaseAmount[:80]
plt.title("Test Data (80%)")
plt.scatter(trainX, trainY)
#plt.show()

#take 20% for testing
testX = pageSpeed[80:]
testY = purchaseAmount[80:]
plt.title("Test Data (20%)")
plt.scatter(testX, testY)
#plt.show()

for grade in range(2,20):
    #Calzar en un polinomio de grado 8 (causar overfitting)
    x = np.array(trainX)
    y = np.array(trainY)
    p4 = np.poly1d(np.polyfit(x, y, grade))
    xp = np.linspace(0, 7, 100)
    axes = plt.axes()
    axes.set_xlim([0, 7])
    axes.set_ylim([0, 200])
    plt.scatter(x, y)
    plt.plot(xp, p4(xp), c='r')
    #plt.show()

    #Visualir el test data. Como vemos no es una buena estimación el polinomio de grado 8
    testx = np.array(testX)
    testy = np.array(testY)
    p4 = np.poly1d(np.polyfit(x, y, grade))
    xp = np.linspace(0, 7, 100)
    axes = plt.axes()
    axes.set_xlim([0, 7])
    axes.set_ylim([0, 200])
    plt.scatter(testx, testy)
    plt.plot(xp, p4(xp), c='r')
    #plt.show()

    #Vamos a calcular nuestro r2, vemos que efectivamente es muy bajo
    r2 = r2_score(testy, p4(testx))
    print "Grade: " + str(grade)
    print "Test: " + str(r2)

    #aunque calce mejor con nuestro training set no es bueno
    r2 = r2_score(trainY, p4(trainX))
    print "training: " + str(r2)
