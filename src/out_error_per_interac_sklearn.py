from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

#x = np.random.random((100, 10))
#y = np.random.random((100, 5))

x = np.arange(0, 2 * np.pi, 0.05).reshape(-1, 1)
y = np.ravel(np.sin(x) * np.cos(2 * x))
len_x = len(x)
len_y = len(y)

#reg = MLPRegressor(hidden_layer_sizes=(5,5), activation='tanh', max_iter=500, solver='lbfgs')
#reg.fit(x, y)
mse = []


reg = MLPRegressor(hidden_layer_sizes=(5,5), activation='tanh', solver='lbfgs', max_iter=1, warm_start=True)
for i in range(1, 500):
    reg.fit(x, y)
    output = reg.predict(x)
    mse.append(mean_squared_error(y, output))
    #print('mse: ', mse)


#print('First fit coef_:\n', reg.coefs_)
#print(reg.loss_)
#plt.plot(x, y, 'b')
#plt.plot(x, output, 'r')
plt.plot(mse)
plt.show()

#reg = MLPRegressor(hidden_layer_sizes=(2), solver='lbfgs', verbose=False, max_iter=1, warm_start=True)
#print('First fit coef_:\n', reg.coefs_)
#reg.fit(x, y)
#print('second fit coef_:\n', reg.coefs_)
#print('\n')
#print(reg.loss_)

#reg = MLPRegressor(hidden_layer_sizes=(2), solver='lbfgs', verbose=False, max_iter=1, warm_start=True)
#print('second fit coef_:\n', reg.coefs_)
#reg.fit(x, y)
#print('third fit coef_:\n', reg.coefs_)
#print('\n')
#print(reg.loss_)

# reg = MLPRegressor(solver='lbfgs', verbose=False, max_iter=1, warm_start=True)
# reg.fit(x, y)
# #print('\n')
# print(reg.loss_)
# reg = MLPRegressor(solver='lbfgs', verbose=False, max_iter=1, warm_start=True)
# reg.fit(x, y)
# #print('\n')
# print(reg.loss_)