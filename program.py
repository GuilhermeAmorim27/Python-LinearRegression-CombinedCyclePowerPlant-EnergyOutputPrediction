import numpy as np
from sklearn.model_selection import train_test_split

X = []
Y = []

#Coletando os dados do arquivo#
for line in open("data\Folds5x2_pp.csv"):
    x1, x2, x3, x4, y = line.split(',')
    X.append([1, float(x1), float(x2), float(x3), float(x4)])
    Y.append(float(y))

#Convertendo os arrays para numpy que são bem mais praticos#
X = np.array(X)
Y = np.array(Y)

#Realizando o treinamento#
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, train_size=0.80)
w = np.linalg.solve(X_train.T.dot(X_train), X_train.T.dot(Y_train))
y_hat = X_test.dot(w)

#Calculando o coeficiente de determinação#
nume = Y_test - y_hat
deno = Y_test - Y_test.mean()
r2 = 1 - nume.dot(nume) / deno.dot(deno)
print(f"A correlação entre os dados previstos e os dados originais é de {r2 * 100:.2f}%")

#Utilizando os pesos para prever manualmente os valores de 21.6, 62.52, 1017.23, 67.87 para 453.28#
original_value = 453.28
prediction = w[0] + 21.6 * w[1] + 62.52 * w[2] + 1017.23 * w[3] + 67.87 * w[4]
print(f'A previsão com base nos dados 21.6, 62.52, 1017.23, 67.87 é {prediction:.2f}')
print(f"A diferença entre valor original {original_value} e a previsão {prediction:.2f} é de {original_value - prediction:.2f}")

