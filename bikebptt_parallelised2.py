import tensorflow as tf
from tensorflow import keras
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from datetime import datetime
from dateutil.relativedelta import relativedelta
plt.ion()
import argparse

VALIDATION = False

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    #try:
    # Currently, memory growth needs to be the same across GPUs
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')


parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--trialname', type=str, default="trial_1")
parser.add_argument('--with_psi_restriction', type=int,default=1)
parser.add_argument('--use_tanh', type=int, default=1)
parser.add_argument('--goal',type=int, default=1)
parser.add_argument('--core',type=int,default=1)
parser.add_argument('--test',type=str,default='psiRemoved')
colors = ['red','blue','green','orange','black','yellow','purple','pink']

#import win32api, win32con, win32process
'''
def setaffinity(mask):
    pid  = win32api.GetCurrentProcessId()
    handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
    win32process.SetProcessAffinityMask(handle,mask)
'''
args = parser.parse_args()
testt = str(args.test)
print("test: "+testt)
mask = int(args.core)
#setaffinity(mask)
trial_name = str(args.trialname)
use_tanh = bool(args.use_tanh)
goal = bool(args.goal)
with_psi_restriction = bool(args.with_psi_restriction)
print(goal)
print(with_psi_restriction)
prinit = True
tf.keras.backend.set_floatx('float64')
noise = 0
graphical = True
save = False
b = 50
'''
theta = handle bar angle
thetadot = velocity of handle bar angle
omega = bike angle
omegadot = velocity of bike angle
'''
display = 5
action_is_theta = True
maximum_dis = 0.02 #0.02
maximum_torque = 2.
batch_size = 10
action_space = 2 if action_is_theta else 1
num_hidden_units = [24,24]
trajectory_length = 100 # This is the length of each "chunk" of trajectory.  Fix this at 50 or 100.
pseudo_trajectory_length=600 # TODO: mahrad, leave this as 600 for now while the colors are debugged.

pseudo_batch_size=batch_size 
if pseudo_trajectory_length>trajectory_length:
    assert pseudo_trajectory_length%trajectory_length==0
    batch_size*=pseudo_trajectory_length//trajectory_length
    #pseudo_trajectory_length=args.pseudo_trajectory_length
else:
    pseudo_trajectory_length=trajectory_length



max_iterations = int(20000)
learning_rate = 0.001
print_time = 50
## BIKE STATS
# Units in meters and kilograms
c = 0.66  # Horizontal distance between point where front wheel touches ground and centre of mass
d_cm = 0.30  # Vertical distance between center of mass and cyclist
h = 0.94  # Height of center of mass over the ground
l = tf.constant(1.11,tf.float64)  # Distance between front tire and back tire at the point where they touch the ground.
m_c = 15.0  # mass of bicycle
m_d = 1.7  # mass of tire
m_p = 60.0  # mass of cyclist
r = 0.34  # radius of tire
v = 10.0 / 3.6  # velocity of the bicycle in m / s 2.7
goal_rsqrd = 1.0
# Useful Precomputations
m = m_c + m_p
inertia_bc = (13. / 3) * m_c * h ** 2 + m_p * (h + d_cm) ** 2  # inertia of bicycle and cyclist
inertia_dv = (3. / 2) * (m_d * (r ** 2))  # Various inertia of tires
inertia_dl = .5 * (m_d * (r ** 2))  # Various inertia of tires
inertia_dc = m_d * (r ** 2)  # Various inertia of tires
sigma_dot = float(v) / r
# Simulation constants
gravity = 9.82
delta_time = 0.01 #0.01 # 0.054 m forward per delta time
randomised_goal_position = False
randomised_state = True
# If omega exceeds +/- 12 degrees, the bicycle falls.
omega_range = np.array([[-np.pi * 12 / 180, np.pi * 12 / 180]] * batch_size)  # 12 degree in SI units.
theta_range = np.array([[-np.pi / 2, np.pi / 2]] * batch_size)
psi_range = np.array([[-np.pi, np.pi]] * batch_size)
rando = 0.0
yg = 60.
xg = 0.
filename = str(args.trialname) + "_with_psi_restriction_"+str(with_psi_restriction)+"_randomised_state_"+str(randomised_state)+"_goal_"+str(goal)+"_test_"+testt+"_tanh_"+str(use_tanh)
goal_position = tf.cast(
    (np.random.uniform(low= -50, high= 50, size=(batch_size, 2))) * (1 if randomised_goal_position else 0),
    tf.float64)
if not randomised_goal_position:
    goal_position += [[xg,yg]]*batch_size


def safe_divide(tensor_numerator, tensor_denominator):
    # attempt to avoid NaN bug in tf.where: https://github.com/tensorflow/tensorflow/issues/2540
    safe_denominator = tf.where(tf.not_equal(tensor_denominator, tf.zeros_like(tensor_denominator,tf.float64)), tensor_denominator,
                                tensor_denominator + 1)
    return tensor_numerator / safe_denominator
def reset():
    # Lagoudakis (2002) randomizes the initial state "arout the
    # equilibrium position"
    if randomised_state:
        theta = np.random.normal(0, 1,size= (batch_size,1)) * np.pi / 180
        omega = np.random.normal(0, 1,size=(batch_size,1)) * np.pi / 180
        thetad = np.zeros((batch_size,1))
        omegad = np.zeros((batch_size,1))
        omegadd = np.zeros((batch_size,1))
        #initial_state=tf.concat([tf.constant( (np.random.rand(batch_size,2)-0.5)*(0 if randomise_food_location else 8), tf.float32), tf.zeros([batch_size, state_dimension-2],tf.float32)],axis=1)
        xb = np.random.uniform(-60,60,(batch_size,1))
        #xb = np.where(np.logical_and(xb >= 0, xb < 20) , xb + 20, xb)
        #xb = np.where(np.logical_and(xb <= 0, xb < -20), xb-20, xb)
        #yb =  np.random.uniform(-60,60,(batch_size,1))
        #yb = np.where(np.logical_and(yb >= 0, yb < 20), yb + 20, yb)
        #yb = np.where(np.logical_and(yb <= 0, yb < -20), yb - 20, yb)
        yb = np.zeros((batch_size, 1), np.float64)
        xf = xb + (np.random.rand(batch_size,1) * l - 0.5 * l)/2 #halved it for psi
        yf = np.sqrt(l ** 2 - (xf - xb) ** 2) + yb
        psi = np.arctan((xb - xf) / (yf - yb))
        psig = psi - np.arctan(safe_divide((xb - xg), yg - yb))
        init_state = np.concatenate([omega, omegad, omegadd, theta, thetad, xf, yf, xb, yb, psi, psig, np.zeros((batch_size,1))], axis=1).astype(np.float64)

    else:
        theta = thetad = omega = omegad = omegadd = xf = yf = xb = yb = np.zeros((batch_size,1))
        yf = yf +l
        psi = np.arctan((xb - xf) / (yf - yb))
        psig = psi - np.arctan(safe_divide((xb - xg), yg - yb))
        init_state = np.concatenate(
            [omega, omegad, omegadd, theta, thetad, xf, yf, xb, yb, psi, psig, np.zeros((batch_size, 1))], axis=1).astype(np.float64)

    # omega, omega_dot, omega_ddot, theta, theta_dot, x_f, y_f, x_b, y_b, psi,psig, timestep
    return init_state


# STATE initialisation
# omega, omega_dot, omega_ddot, theta, theta_dot, x_f, y_f, x_b, y_b, psi,psig, timestep
state_dimension = 12  # omega, omega_dot, omega_ddot, theta, theta_dot, x_f, y_f, x_b, y_b, psi,psig, timestep
initial_state = reset()


def step(state, action, trajectories_terminated, corruption_to_physics_model=None):
    # Unpack the state and actions.
    # -----------------------------
    action = tf.cast(action, tf.float64)
    s = tf.cast(state, tf.float64)
    # omega, omega_dot, omega_ddot, theta, theta_dot, x_f, y_f, x_b, y_b, psi,psig, timestep
    omega = s[:, 0]
    omegad = s[:,1]
    theta = s[:, 3]
    thetad = s[:, 4]  # theta - handle bar, omega - angle of bicycle to verticle psi = bikes angle to the yaxis
    xf = s[:, 5]
    yf = s[:, 6]
    xb = s[:, 7]
    yb = s[:, 8]
    psi = s[:, 9]
    #psi = tf.zeros_like(psi,tf.float64)
    #psig = s[:, 10]
    timestep = s[:, 11]
    # store a last states
    last_pos = s[:, 5:6]
    last_xf = xf
    last_yf = yf
    #last_omega = omega
    #last_psig = psig
    T = action[:, 0] * maximum_torque
    T = tf.where(T > maximum_torque, tf.ones_like(T) * maximum_torque, T)
    T = tf.where(T < -maximum_torque, tf.ones_like(T) * -maximum_torque, T)
    d = action[:, 1] * maximum_dis
    d = tf.where(d > maximum_dis, tf.ones_like(d) * maximum_dis, d)
    d = tf.where(d < -maximum_dis, tf.ones_like(d) * -maximum_dis, d)
    r_f = tf.where(theta == 0., tf.constant(1.e8, tf.float64), safe_divide(l, tf.abs(tf.sin(theta))))
    r_b = tf.where(theta == 0., tf.constant(1.e8, tf.float64), safe_divide(l, tf.abs(tf.tan(theta))))
    r_cm = tf.where(theta == 0., tf.constant(1.e8, tf.float64),
                    tf.sqrt((l - c) ** 2 + (safe_divide(tf.pow(l, 2), (tf.pow(tf.tan(theta), 2))))))

    phi = omega + tf.atan(d / h)

    # Equations of motion.
    # --------------------
    # Second derivative of angular acceleration:
    omegadd = 1 / inertia_bc * (m * h * gravity * tf.sin(phi)
                                   - tf.cos(phi) * (inertia_dc * sigma_dot * thetad
                                                       + tf.sign(theta) * (v ** 2) * (
                                                               m_d * r * (1.0 / r_f + 1.0 / r_b)
                                                               + m * h / r_cm)))
    thetadd = (T - inertia_dv * sigma_dot * omegad) / inertia_dl

    # Integrate equations of motion using Euler's method.
    # ---------------------------------------------------
    # yt+1 = yt + yd * dt.
    # Must update omega based on PREVIOUS value of omegad.
    df = delta_time
    omegad += omegadd * df
    omega += omegad * df
    thetad += thetadd * df
    theta += thetad * df

    # Handlebars can't be turned more than 80 degrees.
    theta = tf.where(theta > 1.3963, tf.ones_like(theta) * 1.3963, theta)
    theta = tf.where(theta < -1.3963, tf.ones_like(theta) * -1.3963, theta)

    # Wheel ('tyre') contact positions.
    # ---------------------------------

    # Front wheel contact position.
    front_term = psi + theta + tf.sign(psi + theta) * tf.asin(v * df / (2. * r_f))
    back_term = psi + tf.sign(psi) * tf.asin(v * df / (2. * r_b))
    xf += v*df * -tf.sin(front_term)
    yf += v*df *tf.cos(front_term)
    xb += v*df *-tf.sin(back_term)
    yb += v*df *tf.cos(back_term)
    # Preventing numerical drift.
    # ---------------------------
    # Copying what Randlov did.
    current_wheelbase = tf.sqrt((xf - xb) ** 2 + (yf - yb) ** 2)
    relative_error = l / current_wheelbase - 1.0
    xb = tf.where(tf.abs(current_wheelbase - l) > 0.01,xb +(xb - xf) * relative_error, xb)
    yb = tf.where(tf.abs(current_wheelbase - l) > 0.01,yb+(yb - yf) * relative_error, yb)
    # Update heading, psi.
    # --------------------
    delta_y = yf - yb
    delta_yg = goal_position[:, 1] - yb
    psi = tf.where(tf.logical_and(xf == xb, delta_y < 0.0), tf.cast(math.pi, tf.float64),
                   tf.where((delta_y > 0.0),
                            tf.atan(safe_divide((xb - xf), delta_y)),
                            tf.sign(xb - xf) * 0.5 * math.pi - tf.atan(safe_divide(delta_y, (xb - xf)))))

    psig = tf.where(tf.logical_and(xf == xb, delta_yg < 0.0), psi - math.pi,
                    tf.where((delta_y > 0.0),
                             psi - tf.atan(safe_divide((xb - goal_position[:, 0]), delta_yg)),
                             psi - tf.sign(xb - goal_position[:, 0]) * 0.5 * math.pi - tf.atan(
                                 safe_divide(delta_yg, (xb - goal_position[:, 0])))))

    omega = tf.reshape(omega, (batch_size, 1))
    omega_dot = tf.reshape(omegad, (batch_size, 1))
    omega_ddot = tf.reshape(omegadd, (batch_size, 1))
    theta = tf.reshape(theta, (batch_size, 1))
    theta_dot = tf.reshape(thetad, (batch_size, 1))
    psig = tf.reshape(psig, (batch_size, 1))
    current_pos = tf.concat([tf.reshape(xf, [batch_size,1]),tf.reshape(yf, [batch_size,1])], axis=1)
    pos_d = current_pos - last_pos
    goal_displacement = goal_position - current_pos
    goal_dist = tf.sqrt(tf.reduce_sum(tf.pow(goal_displacement, 2)))
    goal_displacement_normalised = safe_divide(goal_displacement, goal_dist)
    x_d = xf - last_xf
    y_d = yf - last_yf
    goal_displacement_x = goal_position[:, 0] - xf
    goal_displacement_y = goal_position[:, 1] - yf
    goal_dist = tf.sqrt(tf.pow(goal_displacement_x, 2) + tf.pow(goal_displacement_y, 2))
    goal_displacement_normalised_x = safe_divide(goal_displacement_x, goal_dist)  # constructing a unit vector here.  TODO: need to protect against division by zero here somehow, perhaps?
    goal_displacement_normalised_y = safe_divide(goal_displacement_y, goal_dist)
    if goal:
        r_t = tf.reduce_sum(pos_d * goal_displacement_normalised,axis=1)
        r_t = x_d * goal_displacement_normalised_x + y_d * goal_displacement_normalised_y  # this is a dot product
    else:
        y_d = yf - last_yf
        r_t = y_d
    #r_t = (xf - initial_state[:,5] / goal_position[:,0] - initial_state[:,5]) +  (yf - initial_state[:,6] / goal_position[:,1] - initial_state[:,6])
    # new_state = np.array([omega, omega_dot, omega_ddot, theta, theta_dot])
    x_f = tf.reshape(xf, (batch_size, 1))
    y_f = tf.reshape(yf, (batch_size, 1))
    x_b = tf.reshape(xb, (batch_size, 1))
    y_b = tf.reshape(yb, (batch_size, 1))
    psi = tf.reshape(psi, (batch_size, 1))
    r_t = tf.reshape(r_t, (batch_size, 1))
    timestep = tf.reshape(timestep, (batch_size, 1))
    timestep += 1.
    trajectories_terminating = tf.logical_or(timestep >= pseudo_trajectory_length, tf.abs(omega) > math.pi/9)
    #trajectories_terminating = timestep >= trajectory_length
    trajectories_terminating = tf.reshape(trajectories_terminating, [batch_size, ])
    timestep = tf.reshape(timestep, (batch_size, 1))
    # omega, omega_dot, omega_ddot, theta, theta_dot, x_f, y_f, x_b, y_b, psi,psig, timestep
    new_state = tf.concat([omega, omega_dot, omega_ddot, theta, theta_dot, x_f, y_f, x_b, y_b, psi, psig, timestep],
                          axis=1)
    penalty_handle = flat_bottomed_barrier_function(tf.abs(theta), 1.3963 * 0.9, 8)
    penalty_angle = flat_bottomed_barrier_function(tf.abs(omega), math.pi / 15, 8)
    penalty_psi = 0
    penalty_psig = 0
    #penalty_psi = flat_bottomed_barrier_function(tf.abs(psig), math.pi / 2, 8)* (1 if with_psi_restriction else 0)
    #penalty_psig = flat_bottomed_barrier_function(tf.abs(psig), math.pi / 2, 8)* (1 if goal else 0)
    #remove psi psig in reward
    if testt == "all":
        reward = -tf.tanh(penalty_psi+penalty_angle+penalty_handle)+r_t
    elif testt == "psiRemoved":
        if use_tanh:
            reward = -tf.tanh(penalty_angle+penalty_handle)+r_t
        else:
            reward = -1*(penalty_angle+penalty_handle) + r_t
    elif testt == "angleRemoved":
        reward = -tf.tanh(penalty_psi+penalty_handle)+r_t
    elif testt == "handleRemoved":
        reward = -tf.tanh(penalty_psi+penalty_angle)+r_t
    else:
        reward = -tf.tanh(penalty_psi+penalty_angle+penalty_handle)+r_t
    return [reward, new_state, trajectories_terminating]


def flat_bottomed_barrier_function(x, k_width, k_power):
    return tf.pow(tf.maximum(x / (k_width * 0.5) - 1, 0), k_power)


def is_at_goal(position, goal_loc, goal_rsqrd):
    xy = np.concatenate([[position[:, 0]], [position[:, 1]]], axis=1)
    dist_btw_goal = np.sqrt(max(0., tf.reduce_sum((xy - goal_loc) ** 2) - goal_rsqrd))
    return dist_btw_goal


class model(keras.Model):
    def __init__(self):
        super(model, self).__init__()
        self.neural_layers = []

        for hidden in num_hidden_units:
            self.neural_layers.append(keras.layers.Dense(hidden, activation="tanh",kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
bias_initializer=keras.initializers.Zeros()))
        self.neural_layers.append(keras.layers.Dense(action_space, name='output', activation="tanh",
                                                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
                                                     bias_initializer=keras.initializers.Zeros()))
    @tf.function
    def call(self, input):
        x = input
        for layer in self.neural_layers:
            y = layer(x)
            x = tf.concat([x, y], axis=1)
        return y


keras_action_network = model()
if VALIDATION == True:
    keras_action_network.load_weights("./checkpoints/my_checkpoint")

def evaluate_final_state(state):
    return tf.zeros_like(state[:, 0])



def expand_trajectories(start_state):
    total_rewards = tf.constant(0.0, dtype=tf.float64, shape=[batch_size])
    actions = tf.zeros((batch_size, action_space), tf.float64)
    action_history = tf.expand_dims(actions, axis=0)
    action_list = [actions]
    trajectory_list = [start_state]
    trajectories_terminated = tf.cast(tf.zeros_like(start_state[:, 0]), tf.bool)
    # TODO mahrad, this function does not return the actions history correctly, can you fix this?
    state = start_state
    # build main graph.  This is a long graph with unrolled in time for trajectory_length steps.  Each step includes one neural network followed by one physics-model
    for t in range(trajectory_length):
        converted_state = converter(state)
        prevaction = keras_action_network(converted_state)
        action = tf.reshape(prevaction, (batch_size, action_space))
        [rewards, n_state, trajectories_terminating] = step(state, action, trajectories_terminated)
        state = tf.where(tf.expand_dims(trajectories_terminated, 1), state, n_state)
        T = prevaction[:, 0] * maximum_torque
        T = tf.where(T > maximum_torque, tf.ones_like(T) * maximum_torque, T)
        T = tf.where(T < -maximum_torque, tf.ones_like(T) * -maximum_torque, T)
        d = prevaction[:, 1] * maximum_dis
        d = tf.where(d > maximum_dis, tf.ones_like(d) * maximum_dis, d)
        d = tf.where(d < -maximum_dis, tf.ones_like(d) * -maximum_dis, d)
        T = tf.reshape(T,(batch_size,1))
        d = tf.reshape(d,(batch_size,1))
        prevaction = tf.concat([T, d], axis=1)
        action_list.append(prevaction)
        rewards = tf.reshape(rewards, (batch_size,))
        total_rewards += tf.where(trajectories_terminated, tf.zeros_like(rewards), rewards)
        total_rewards += tf.where(tf.logical_and(trajectories_terminating, tf.logical_not(trajectories_terminated)),
                                  evaluate_final_state(state), tf.zeros_like(rewards))
        trajectories_terminated = tf.logical_or(trajectories_terminated, trajectories_terminating)
        trajectory_list.append(state)
    action_history = tf.stack(action_list, axis=0)
    trajectory = tf.stack(trajectory_list, axis=0)
    average_total_reward = tf.reduce_mean(total_rewards)
    return [average_total_reward, trajectory, action_history,trajectories_terminated,state]

'''
# Mahrad, I've replaced the while_loop by a for loop now; the for_loop is easer to read and edit.
def expand_trajectories_with_while_loop(start_state):
    total_rewards = tf.constant(0.0, dtype=tf.float64, shape=[batch_size])
    actions = tf.zeros((batch_size, action_space), tf.float64)
    action_history = tf.expand_dims(actions, axis=0)
    trajectory = tf.expand_dims(start_state, axis=0)
    trajectories_terminated = tf.cast(tf.zeros_like(start_state[:, 0]), tf.bool)
    state = start_state
    # build main graph.  This is a long graph with unrolled in time for trajectory_length steps.  Each step includes one neural network followed by one physics-model
    [state, total_rewards, trajectory, trajectories_terminated,
     action_history] = tf.while_loop(while_loop_cond, while_loop_body, (
        state, total_rewards, trajectory, trajectories_terminated, action_history),
                                     shape_invariants=(
                                         state.get_shape(), total_rewards.get_shape(),
                                         tf.TensorShape([None, state.get_shape()[0], state_dimension]),
                                         trajectories_terminated.get_shape(),
                                         tf.TensorShape([None, batch_size, action_space])))
    average_total_reward = tf.reduce_mean(total_rewards)

    return [average_total_reward, trajectory, action_history,trajectories_terminated,state]


def while_loop_cond(state, total_rewards, trajectory, trajectories_terminated,
                    action_history):
    return tf.logical_not(tf.reduce_all(trajectories_terminated))

def while_loop_body(state, total_rewards, trajectory, trajectories_terminated,
                    action_history):
    converted_state = converter(state)
    prevaction = keras_action_network(converted_state)
    action = tf.reshape(prevaction, (batch_size, action_space))
    [rewards, n_state, trajectories_terminating] = step(state, action, trajectories_terminated)
    state = tf.where(tf.expand_dims(trajectories_terminated, 1), state, n_state)
    #T = prevaction[:, 0] * maximum_torque
    #T = tf.where(T > maximum_torque, tf.ones_like(T) * maximum_torque, T)
    #T = tf.where(T < -maximum_torque, tf.ones_like(T) * -maximum_torque, T)
    #d = prevaction[:, 1] * maximum_dis
    #d = tf.where(d > maximum_dis, tf.ones_like(d) * maximum_dis, d)
    #d = tf.where(d < -maximum_dis, tf.ones_like(d) * -maximum_dis, d)
    #T = tf.reshape(T,(batch_size,1))
    #d = tf.reshape(d,(batch_size,1))
    #prevaction = tf.concat([T, d], axis=1)
    rewards = tf.reshape(rewards, (batch_size,))
    total_rewards += tf.where(trajectories_terminated, tf.zeros_like(rewards), rewards)

    total_rewards += tf.where(tf.logical_and(trajectories_terminating, tf.logical_not(trajectories_terminated)),
                              evaluate_final_state(state), tf.zeros_like(rewards))
    trajectories_terminated = tf.logical_or(trajectories_terminated, trajectories_terminating)
    trajectory = tf.concat([trajectory, tf.expand_dims(state, axis=0)], axis=0)
    #action_history = tf.concat([action_history, tf.expand_dims(prevaction, axis=0)], axis=0)
    #total_rewards = tf.reshape(total_rewards, [batch_size, ])
    #trajectories_terminated = tf.reshape(trajectories_terminated, [batch_size, ])
    return state, total_rewards, trajectory, trajectories_terminated, action_history'''


def compass_calculation(xy):
    direction = goal_position - xy
    return tf.tanh(direction)

def converter(state):
    # omega, omega_dot, omega_ddot, theta, theta_dot, x_f, y_f, x_b, y_b, psi,psig, timestep
    omega = state[:, 0]
    omega_dot = state[:, 1]
    omega_ddot = state[:, 2]
    theta = state[:, 3]
    theta_dot = state[:, 4]  # theta - handle bar, omega - angle of bicycle to verticle
    x_f = state[:, 5]
    y_f = state[:, 6]
    x_b = state[:, 7]
    y_b = state[:, 8]
    psi = state[:, 9]
    psig = state[:, 10]
    timestep = state[:, 11]
    x_f = tf.reshape(x_f, (batch_size, 1))
    y_f = tf.reshape(y_f, (batch_size, 1))
    x_b = tf.reshape(x_b, (batch_size, 1))
    y_b = tf.reshape(y_b, (batch_size, 1))
    psi = tf.reshape(psi, (batch_size, 1))
    omega = tf.reshape(omega, (batch_size, 1))
    omega_dot = tf.reshape(omega_dot, (batch_size, 1))
    omega_ddot = tf.reshape(omega_ddot, (batch_size, 1))
    theta = tf.reshape(theta, (batch_size, 1))
    theta_dot = tf.reshape(theta_dot, (batch_size, 1))
    psig = tf.reshape(psig, (batch_size, 1))
    xy = tf.concat([x_f, y_f], axis=1)
    timestep = tf.reshape(timestep, (batch_size,1))
    direction = compass_calculation(xy)
    omega_visible = tf.tanh(omega*10)
    omega_dot = tf.tanh(omega_dot)
    omega_ddot = tf.tanh(omega_ddot)
    theta_dot = tf.tanh(theta_dot)
    theta = tf.tanh(theta / (math.pi / 4))
    #yaw = tf.tanh(psi / (math.pi / 4))
    if goal:
        converted_state = tf.concat([omega_visible,omega_dot, theta,theta_dot,tf.sin(psig),tf.cos(psig)], axis=1)
    else:
        converted_state = tf.concat([omega_visible,omega_dot, theta,theta_dot,tf.sin(psi),tf.cos(psi)], axis=1)
    return converted_state



opt = keras.optimizers.Adam(learning_rate)
reward_history = []
iterations = []


def add_arrow_to_line2D(axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8], arrowstyle='-|>', arrowsize=1, transform=None):
    if not isinstance(line, mlines.Line2D):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line.get_xdata(), line.get_ydata()
    arrow_kw = {
        "arrowstyle": arrowstyle,
        "mutation_scale": 10 * arrowsize,
    }
    color = line.get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        raise NotImplementedError("multicolor lines not supported")
    else:
        arrow_kw['color'] = color
    linewidth = line.get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multiwidth lines not supported")
    else:
        arrow_kw['linewidth'] = linewidth
    if transform is None:
        transform = axes.transData
    arrows = []
    for loc in arrow_locs:
        s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n], y[n])
        arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        p = mpatches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform,
            **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)
    return arrows


def static_graphics():
    fig, ((ax_omega, ax_theta), (ax_trajectory, ax_rewardOverTime), (ax_actionT, ax_psi), (ax_actiond, ax_timestep)) = plt.subplots(nrows=4, ncols=2, figsize=(10, 10))
    # display.clear_output(wait=True)
    # fig=plt.figure(figsize=[12.4, 4.8])
    fig.tight_layout(pad=5.0)
    # ax_omega = omega
    pad = 10
    ax_omega.axis([0, pseudo_trajectory_length,(-math.pi/15)* 180 / math.pi - pad , (math.pi/15)* 180 / math.pi +pad])
    ax_omega.set(xlabel='timestep', ylabel='Bike Roll value in Degrees')
    ax_omega.set_title('(Bike Roll).')
    for traj in range(batch_size):
        d = [0]  # trajectory[:,traj,2]*180/math.pi
        ax_omega.plot(d)
    ax_omega.grid()
    # ax2 = theta
    ax_theta.axis([0, pseudo_trajectory_length, -80-pad, 80+pad])
    ax_theta.set(xlabel='timestep', ylabel='Bike handle value in Degrees')
    ax_theta.set_title('(Bike Handle).')
    for traj in range(batch_size):
        trajectory_x_coord = [0]  # trajectory[:,traj,0]
        ax_theta.plot(trajectory_x_coord)
    ax_theta.grid()
    # ax3 = Agent moving in the field
    ax_trajectory.axis([-b,b,-b,b])
    ax_trajectory.set(xlabel='x', ylabel='y')
    ax_trajectory.set_title('Bike Trajectory.')
    x_f = initial_state[:, 5]
    y_f = initial_state[:, 6]
    ax_trajectory.plot(goal_position[:, 0], goal_position[:, 1], color='green', marker='o')
    ax_trajectory.grid()
    # ax4 = reward over time.
    ax_rewardOverTime.axis([0, max_iterations, -1-0.5, 1+0.5])
    ax_rewardOverTime.set(xlabel='Iteration', ylabel='Reward')
    ax_rewardOverTime.set_title('Reward over Iteration')
    for traj in range(batch_size):
        trajectory_x_coord = [0]  # trajectory[:,traj,0]
        ax_rewardOverTime.plot(trajectory_x_coord)
    ax_rewardOverTime.grid()
    # ax5 = action the agent takes
    ax_actionT.axis([0, trajectory_length, -2-0.5, 2+0.5])
    ax_actionT.legend(loc="upper right")
    ax_actionT.set(xlabel='timestep', ylabel='torque')
    ax_actionT.set_title('Trajectory torque')
    for traj in range(batch_size):
        trajectory_x_coord = [0]  # trajectory[:,traj,0]
        ax_actionT.plot(trajectory_x_coord)
    ax_actionT.grid()
    # ax6 = agent psi
    ax_psi.axis([0, pseudo_trajectory_length, -180-pad, 180+pad])
    ax_psi.set(xlabel='timestep', ylabel='psi')
    ax_psi.set_title('Bike direction Psi')
    for traj in range(batch_size):
        trajectory_x_coord = [0]  # trajectory[:,traj,0]
        ax_psi.plot(trajectory_x_coord)
    ax_psi.grid()
    #ax7 = agent d
    ax_actiond.axis([0,pseudo_trajectory_length,-maximum_dis-0.01,maximum_dis+0.01])
    ax_actiond.set(xlabel= 'timestep', ylabel = 'displacement')
    ax_actiond.set_title('The centre of mass displacement')
    for traj in range(batch_size):
        trajectory_x_coord = [0]  # trajectory[:,traj,0]
        ax_actiond.plot(trajectory_x_coord)

    ax_actiond.grid()
    #ax8 timestep
    ax_timestep.axis([0,max_iterations,0-pad,trajectory_length+pad])
    for traj in range(max_iterations):
        trajectory_x_coord = [0]  # trajectory[:,traj,0]
        ax_timestep.plot(trajectory_x_coord)
    ax_timestep.set_title('Timestep over Iteration') # TODO Mahrad, retitle this graph to "Balancing Duration", and set x-axis label "Iteration", and y-axis label "time steps"
    ax_timestep.grid()
    plt.draw()
    plt.pause(0.001)
    return [fig, ax_omega, ax_theta, ax_trajectory, ax_rewardOverTime, ax_actionT, ax_psi,ax_actiond,ax_timestep]
def dynamic_graphics(trajectory,stats):
    # omega, omega_dot, omega_ddot, theta, theta_dot, x_f, y_f, x_b, y_b, psi,psig, timestep
    fig,ax_omega, ax_theta,ax_trajectory, ax_rewardOverTime,ax_actionT, ax_psi, ax_actiond , ax_timestep=stats
    omega = trajectory[:, :, 0]
    omega_dot = trajectory[:, :, 1]
    omega_ddot = trajectory[:, :, 2]
    theta = trajectory[:, :, 3]
    theta_dot = trajectory[:, :, 4]  # theta - handle bar, omega - angle of bicycle to verticle
    x_f = trajectory[:, :, 5]
    y_f = trajectory[:, :, 6]
    x_b = trajectory[:, :, 7]
    y_b = trajectory[:, :, 8]
    psi = trajectory[:, :, 9]
    time_steps = trajectory[:, :, -1]
    trajectory_omega_history = []
    trajectory_handle_history = []
    trajectory_psi_history = []
    for t in range(len(omega)):
        trajectory_omega_coord = np.mean(omega[t] * 180 / math.pi)
        trajectory_omega_history.append(trajectory_omega_coord)
    for t in range(len(theta)):
        trajectory_handle = np.mean(theta[t] * 180 / math.pi)
        trajectory_handle_history.append(trajectory_handle)
    for t in range(len(psi)):
        trajectory_psi = np.mean(psi[t] * 180 / math.pi)
        trajectory_psi_history.append(trajectory_psi)
    # ax1 = omega
    # ax2 = theta
    # ax3 = Agent moving in the field
    # ax4 = reward over time.
    # ax5 = action the agent takes 1 T
    # ax6 = agent psi
    # ax7 = agent action 2 d
    # ax8 = iteration vs timestep

    col = 0
    col_trajectory = 0
    ax_omega.lines.clear()
    ax_theta.lines.clear()
    ax_trajectory.lines.clear()
    ax_rewardOverTime.lines.clear()
    ax_actionT.lines.clear()
    ax_psi.lines.clear()
    ax_actiond.lines.clear()
    ax_timestep.lines.clear()

    # mahrad, can you make the colours wrap here like done for ax3 below?  Similarly for all graphs, use the same colour rotating logic.
    for t in range(batch_size):
        col = (col + 1) % len(colors)
        ax_omega.plot(time_steps[:,t],omega[:, t] * 180 / math.pi, color=colors[col])
        ax_theta.plot(time_steps[:,t],theta[:,t] * 180 / math.pi, color=colors[col])
        x = x_f[:,t]
        y = y_f[:,t]
        if time_steps[0,t]==0:
            col_trajectory=(col_trajectory+1)%len(colors)
        ax_trajectory.plot(x, y, color=colors[col_trajectory])
        ax_actionT.plot(actions[:, t, 0], color=colors[col])
        ax_psi.plot(time_steps[:, t], psi[:, t] * 180 / math.pi, color=colors[col])

    ax_trajectory.plot(goal_position[t, 0], goal_position[t, 1], color='b', marker='o')
    ax_rewardOverTime.axis([0, max_iterations, min(reward_history) - 10, max(reward_history) + 10])
    ax_rewardOverTime.plot(reward_history, color='red')
    ax_actiond.plot(actions[:, t, 1], color=colors[col])
    d = np.mean(actions[:, :, 1], axis=1)
    ax_timestep.plot(timestep_history, color='red')
    plt.draw()
    plt.pause(0.001)
if graphical:
    stat = static_graphics()
t_a = datetime.now()
t_b = datetime.now()
@tf.function
def dolearn(statea):
    with tf.GradientTape() as t:
        [total_reward, stateb, action_history,trajectories_terminated,last_state] = expand_trajectories(statea)
        cost_ = -tf.reduce_mean(total_reward)
    dgrad = t.gradient(cost_, [statea]+keras_action_network.trainable_weights) # Mahrad this line is new.  It calculates the derivative which we will need to wrap around to the previous chunk of time-steps.
    d_reward_dInputStates= -dgrad[0] # putting an extra minus sign here to cancel out the above minus sign, and get back to positive rewards.
    d_reward_dWeights=dgrad[1:]
    return d_reward_dWeights, d_reward_dInputStates, stateb, total_reward,action_history,trajectories_terminated,last_state

def diff(t_a, t_b):
    t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)
timestep_history = []
action_history = []
initial_state_backup = initial_state.copy()
trajectories_terminated = tf.cast(tf.zeros_like(initial_state[:, 0]), tf.bool)

for iteration in range(max_iterations):
    state1 = initial_state.copy()
    #trajectories_terminated = trajectories_terminated
    d_loss_dWeights, d_reward_dInputStates, trajectory, total_reward,actions,trajectories_terminated,last_state = dolearn(state1)
    trajectory = trajectory.numpy()
    trajectories_terminated=trajectories_terminated.numpy()
    for _ in d_loss_dWeights:
        if np.isnan(_).any():
            print("Nan Grads")
            
    # do trajectory wrap-around hack.
    for i_ in range(1,batch_size):
        if trajectories_terminated[i_-1]:
            # the previous trajectory crashed, so the next trajectory needs to start from the beginning
            initial_state[i_,:]=initial_state_backup[i_,:]
        else:
            # the previous trajectory did not crash, so it is still going! So let the next trajectory start from where the old one left off...
            # (This is to reposition the "time-step" dimension into the batch-size dimension.  We do this because we'll get much better parallelism
            # on the graphics card / inner c++ loops if we do this.  However it's only approximate - there might be tiny gaps appearing in a set of trajectories
            # which are linked together like this)
            initial_state[i_,:]=trajectory[-1,i_-1,:] # copy over every state variable so that next trajectory starts where old one ended.  
    #total_reward = total_reward + (-1 *(trajectory_length - trajectory[-1,:,-1]))
    if VALIDATION == False:
        opt.apply_gradients(zip(d_loss_dWeights, keras_action_network.trainable_weights))
    average_total_reward_stepwise = np.mean(total_reward.numpy())
    reward_history.append(average_total_reward_stepwise)
    timestep_history.append(np.mean(trajectory[-1,:,-1]))
    final_trajectory_steps = trajectory[-1, :, :]
    '''
    for b in range(1,batch_size):
        if not trajectories_terminated[b-1]:
            initial_state[b,:].assign(final_trajectory_steps[b-1,:]+0)
            print("passed")
        else:
            initial_state[b,:].assign( initial_state_backup[b-1,:]+0)
    '''
    if save:
        if iteration %500 == 0:
            #np.save("runs/"+trial_name+"_marker_"+str(iteration)+"_action_history_" + filename + ".npy", np.array(action_history))
            np.save("runs/"+trial_name+"_marker_"+str(iteration)+"_reward_history_" + filename + ".npy", np.array(reward_history))
            np.save("runs/"+trial_name+"_marker_"+str(iteration)+"_step_history_" + filename + ".npy", np.array(timestep_history))
            action_history = []
            reward_history = []
            timestep_history = []
        if iteration %1000 == 0:
            keras_action_network.save_weights("./checkpoints/my_checkpoint")
    if prinit:
        if iteration % print_time == 0:
            t_b = datetime.now()
            dt = t_b - t_a
            print("iteration: ", iteration, "// Average_total_reward_step_wise: ", average_total_reward_stepwise,"Average Step: ",np.mean(trajectory[-1,:,-1]) ,"in steps and ",np.mean(trajectory[-1,:,-1])*delta_time,"in seconds", "time taken from last iter: ",diff(t_a,t_b))
            if save: 
                np.save("runs/last_state_" + filename + ".npy", trajectory)
            t_a = t_b
    if graphical:
        if iteration % print_time == 0:
            dynamic_graphics(trajectory,stat)
if save:
    #stat = static_graphics()
    np.save("runs/" + trial_name + "_marker_" + str(iteration) + "_action_history_" + filename + ".npy",
            np.array(actions.numpy()))
    np.save("runs/" + trial_name + "_marker_" + str(iteration) + "_reward_history_" + filename + ".npy",
            np.array(reward_history))
    np.save("runs/" + trial_name + "_marker_" + str(iteration) + "_step_history_" + filename + ".npy",
            np.array(timestep_history))
    np.save("runs/last_state_"+filename+".npy",trajectory)
    #dynamic_graphics(state, stat)
    #plt.savefig("figure_" + filename + ".jpg")
    keras_action_network.save_weights("./checkpoints/my_checkpoint")

