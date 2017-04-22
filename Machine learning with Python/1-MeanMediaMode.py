# coding=utf-8

#Vamos a crear una data falsa centrada en 27000, con una desviación estándar de 15000 y unos 10000 samples.
#Si calculamos la media, nos debería dar un valor cercano a 27000
import numpy as np
incomes = np.random.normal(27000, 15000, 10000)
print ("Mean: "),
print np.mean(incomes)

#Vamos a segmentar la data entrante en bloques de 50 mediante un histograma (Discretizar en 50 clases)
import matplotlib.pyplot as plt
plt.hist(incomes, 50)
#plt.show()

#Calcular la mediana
print ("Median: "),
print np.median(incomes)

#Ahora agregamos a Donald Trump
incomes = np.append(incomes, 1000000000)
print ("------")
print ("Mean: "),
print np.mean(incomes)
print ("Median: "),
print np.median(incomes)

#Para clacular el Mode vamos a generar 500 edades de personas entre 18 a 90
ages = np.random.randint(18, high=90, size = 500)
print ages
from scipy import stats
print stats.mode(ages)

#Deviación estandar y varianze
print ("Standart deviation"),
print incomes.std()
print ("Variance"),
print incomes.var()
