#import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras import Model
import numpy as np 
import scipy.io
from pyDOE import lhs
import time
import pinn_class_model as PINN
import matplotlib.pyplot as plt

#set upper and lower spatial bounds
lb = np.array([-5.0, 0.0])
ub = np.array([5.0, np.pi/2])

#load data from NLS.mat
data = scipy.io.loadmat('../data/NLS.mat')

#slice and assign data:
t = data['tt'].flatten()[:,None]
x = data['x'].flatten()[:,None]
print(x.shape, t.shape)
Exact = data['uu']
Exact_u = np.real(Exact)
Exact_v = np.imag(Exact)
Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)

#set initial, boundary and colocation number of points
N0 = 50
N_b = 50
N_f = 20000

#create random indices:
idx_x = np.random.choice(x.shape[0], N0, replace=False)
idx_t = np.random.choice(t.shape[0], N_b, replace=False)

#create initial data:
x0 = x[idx_x, :]
t0 = x0*0.0
u0 = Exact_u[idx_x, 0:1]
v0 = Exact_v[idx_x, 0:1]
tb = t[idx_t,:]

#create colocation points:
X_f = lb + (ub - lb) * lhs(2, N_f)

#convert data to tensors:
x0_t0 = tf.convert_to_tensor(np.concatenate((x0, t0), 1), dtype=tf.float32)
xlb_tlb = tf.convert_to_tensor(np.concatenate((0.0*tb + lb[0], tb), 1), dtype=tf.float32)
xub_tlb = tf.convert_to_tensor(np.concatenate((0.0*tb + ub[0], tb), 1), dtype=tf.float32)
xf_tf = tf.convert_to_tensor(X_f, dtype=tf.float32)
u0_tf = tf.convert_to_tensor(u0, dtype=tf.float32)
v0_tf = tf.convert_to_tensor(v0, dtype=tf.float32)

#create test data:
X, T = np.meshgrid(x, t, sparse=False)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact_u.T.flatten()[:,None]
v_star = Exact_v.T.flatten()[:,None]
h_star = Exact_h.T.flatten()[:,None]

#create model instance:
model = PINN.PINN(lb = lb, ub = ub)

#training of model:
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
NUMBER_EPOCHS = 10000

for epoch in range(NUMBER_EPOCHS):
    with tf.GradientTape() as tape:
        current_loss = model.loss_fn(x0_t0, xlb_tlb, xub_tlb, xf_tf, u0_tf, v0_tf)
    dW = tape.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(dW, model.trainable_variables))

    if epoch % 10 == 0:
        print(epoch, current_loss)

#convert to tensor
X_star_tf = tf.convert_to_tensor(X_star, dtype=tf.float32)

#calculate outputs:
predictions = model(X_star_tf)
u_pred = predictions[:,0:1]
v_pred = predictions[:,1:2]
h_pred = tf.sqrt(u_pred**2 + v_pred**2)
u_pred = u_pred.numpy()
v_pred = v_pred.numpy()
h_pred = h_pred.numpy()

#calculate errors:
error_u = np.linalg.norm(u_star - u_pred,2)/np.linalg.norm(u_star,2)
error_v = np.linalg.norm(v_star - v_pred,2)/np.linalg.norm(v_star,2)
error_h = np.linalg.norm(h_star - h_pred,2)/np.linalg.norm(h_star,2)

print("u error: ", error_u)
print("v error: ", error_v)
print("h error: ", error_h)

#plot results for 0.75s and 1s:
index = 75
plt.plot(X_star[0:256,0], h_star[index*256:index*256+256], 'ro', alpha=0.2, label='0.75s actual')
plt.plot(X_star[0:256,0], h_pred[index*256:index*256+256], 'k', label='0.75s pred.')

index = 100
plt.plot(X_star[0:256,0], h_star[index*256:index*256+256], 'bo', alpha=0.2, label='1s actual')
plt.plot(X_star[0:256,0], h_pred[index*256:index*256+256], 'k--', label='1s pred.')

plt.legend()
plt.show()