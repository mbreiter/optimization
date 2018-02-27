import autograd.numpy as np
from autograd import grad, jacobian
from matplotlib import pyplot as plt
from datetime import datetime

# true value for the regression parameters
true_w = np.array([1., 2., 4.])

# prediction function
def y(x,w):
	return (w[0] + w[1]*np.sin(2. * np.pi * x)) / (1 + w[2]*x)

def loss_function(w, dat):
	return 0.5 * np.sum(np.power(y(dat[:,0], w) - dat[:,1], 2))

loss_gradient = grad(loss_function)
loss_hessian = jacobian(loss_gradient)

def generate_data(n):
	x = np.random.uniform(size=n)
	return np.stack((x, y(x, true_w) + np.random.normal(scale=0.1, size=n)), axis=1)

dat = generate_data(n=30)

plt.scatter(x=dat[:,0], y=dat[:,1])
plt.plot(np.linspace(0,1,200), y(np.linspace(0,1,200), true_w))
plt.show()

w = np.array([0., 0., 0.])
step = 0.01
t = 0

print(loss_gradient(w,dat))

starttime = datetime.now()

while np.linalg.norm(loss_gradient(w,dat)) > 0.0001:
	w = w - step*loss_gradient(w,dat)
	t = t+1

	if t%100 == 0:
		print("At iteration {}, loss function is {} and norm of gradient is {}".format(t,loss_function(w,dat),np.linalg.norm(loss_gradient(w,dat))))

print("Final loss function: {}".format(loss_function(w,dat)))
print("Final parameter estimate: {}".format(w))
print("True parameter estimate: {}".format(true_w))

endtime = datetime.now()
print("{} iterations of gradient descent took {}".format(t,endtime - starttime))
