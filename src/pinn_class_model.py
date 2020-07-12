import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras import Model

#create model class
class PINN(Model):
    def __init__(self, lb, ub):
        super(PINN, self).__init__()
        
        #create layers: [2 100 100 100 100 2]
        self.fc1 = Dense(100, input_shape=(None,2), activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.fc2 = Dense(100, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.fc3 = Dense(100, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.fc4 = Dense(100, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.output_layer = Dense(2, activation = 'linear', kernel_initializer = 'glorot_uniform')
        
        self.lb = lb
        self.ub = ub
    
    #call method
    def call(self, x):
        return self.net(x)
    
    #class method which returns u,v and derivatives
    def net_uv(self, x):
        with tf.GradientTape(persistent=True) as tape_1:
            tape_1.watch(x)
            net_output = self.net(x)        
            u = net_output[:,0:1]
            v = net_output[:,1:2]
            
        u_x = tape_1.gradient(u, x)[:,0:1]
        v_x = tape_1.gradient(v, x)[:,0:1]
        
        del tape_1
        return u, v, u_x, v_x
    
    #class method which returns pde residuals
    def net_f_uv(self, x):
        with tf.GradientTape(persistent=True) as tape_2:
            tape_2.watch(x)
            with tf.GradientTape(persistent=True) as tape_3:
                tape_3.watch(x)
                output = self.net(x)
                u = output[:,0:1]
                v = output[:,1:2]
            u_x = tape_3.gradient(u, x)[:,0:1]
            v_x = tape_3.gradient(v, x)[:,0:1]
            u_t = tape_3.gradient(u, x)[:,1:2]
            v_t = tape_3.gradient(v, x)[:,1:2]
        u_xx = tape_2.gradient(u_x, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:,0:1]
        v_xx = tape_2.gradient(v_x, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:,0:1]
                    
        f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
        f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u
        
        del tape_2, tape_3
        return f_u, f_v
    
    #network forward propagation method
    def net(self, x):
        x = 2.0 * (x - self.lb)/(self.ub - self.lb) - 1.0
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.output_layer(self.fc4(x))
    
    #loss function method
    def loss_fn(self, x0_t0, xlb_tlb, xub_tub, xf_tf, u0, v0):
        u0_pred, v0_pred, _, _ = self.net_uv(x0_t0)
        u_lb_pred, v_lb_pred, u_x_lb_pred, v_x_lb_pred = self.net_uv(xlb_tlb)
        u_ub_pred, v_ub_pred, u_x_ub_pred, v_x_ub_pred = self.net_uv(xub_tub)
        f_u_pred, f_v_pred = self.net_f_uv(xf_tf)
        
        loss = tf.reduce_mean(tf.square(u0_pred - u0)) + \
               tf.reduce_mean(tf.square(v0_pred - v0)) + \
               tf.reduce_mean(tf.square(u_lb_pred - u_ub_pred)) + \
               tf.reduce_mean(tf.square(v_lb_pred - v_ub_pred)) + \
               tf.reduce_mean(tf.square(u_x_lb_pred - u_x_ub_pred)) + \
               tf.reduce_mean(tf.square(v_x_lb_pred - v_x_ub_pred)) + \
               tf.reduce_mean(tf.square(f_u_pred)) + \
               tf.reduce_mean(tf.square(f_v_pred))
        return loss