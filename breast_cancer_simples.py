import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)


classificador = Sequential()
classificador.add(Dense(units = 16, activation = 'relu',
                        kernel_initializer = 'random_uniform', input_dim = 30))
classificador.add(Dense(units = 1, activation = 'sigmoid'))
