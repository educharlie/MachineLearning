1) Install pip
  sudo easy_install pip
2) Install Scikit learn
  sudo pip install -U scikit-learn
3)Install pandas
  sudo pip install -U pandas
4) xlrd (Import Excel files)
  sudo pip install xlrd
5) statsmodels
  sudo pip install statsmodels


Read comma separated lines in python, use tuples
(age, income) = "32,120000".split(',')



Curso

1) Revisión de Python
  Listas, Tuplas, Diccionarios
  Fucniones, parametros pueden ser varaibles, funciones, lambdas
  Condiciones, loops,

2) Tipos de datos
  a) Numéricos
    - Datos dscretos
    - Datos contínuos
  b) Categoricos
  c) Ordinarios

3) Estadistica
  a) Mean
    Average
  b) Median
    Ordenar los valores y tomar el valor de la mitad. La media da una mejor apreciación del conjunto de datos que el promedio. (Ej. ingreso familiar promedio en US, los ricos tienden a subir esa media)
  c) Mode
    El valor más común en un conjunto de datos. Mode es usado en datos discretos
  d) Ejemplos en Python
  e) Desviación estandar y varianza
4) PDF (Probability Density Function), PMF (Probability Mass function)
  a) PDF = continuous data (curve)
  b= PMS = discrete data (data set)
5) Percentiles y momentos
  Percentil: valor de 90th percentile indica que que el 90% de los datos son menores que el valor resultante
  Momento: manera de medir la forma de la PDF
    a) primer momento = mean
    b) segundo momento = varianza
    c) tercer momento (skew): mide cuan desbalanceado en la PDF (asimetría)
    4) cuarto momento (kurtosis): Que tan grueso es la cola y que tan agudo es el pico de la PDF. sirve para analizar el grado de concentración que presentan los valores de una variable analizada alrededor de la zona central de la distribución de frecuencias, sin necesidad de generar el gráfico.
