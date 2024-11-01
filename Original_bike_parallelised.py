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
import argparse
VALIDATION = False

tf.keras.backend.set_floatx("float32")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
#PARSER

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--trialname', type=str, default="trial_3")
parser.add_argument('--with_psi_restriction', type=int, default=1)
parser.add_argument('--use_tanh', type=int, default=0)
parser.add_argument('--goal', type=int, default=0)
parser.add_argument('--core', type=int, default=1)
parser.add_argument('--test', type=str, default='psiRemoved')
args = parser.parse_args()
testt = str(args.test)
mask = int(args.core)
trial_name = str(args.trialname)
use_tanh = bool(args.use_tanh)
goal = bool(args.goal)
with_psi_restriction = bool(args.with_psi_restriction)
#AFINITY
use_cpu_afinity = True
if use_cpu_afinity:
    import win32api, win32con, win32process
    def setaffinity(mask):
        pid  = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetProcessAffinityMask(handle,mask)
    setaffinity(mask)
#EXPERIMENT SETTINGS
max_iterations = 150
b = 60
pi = tf.constant(math.pi)
print(pi)
action_is_theta = True
maximum_dis = 0.02  # 0.02
maximum_torque = 2.
batch_size = 10
action_space = 2 if action_is_theta else 1
num_hidden_units = [24, 24]
trajectory_length = 200
pseudo_trajectory_length = 1000
pseudo_batch_size = batch_size

if pseudo_trajectory_length > trajectory_length:
    assert pseudo_trajectory_length % trajectory_length == 0
    batch_size *= pseudo_trajectory_length // trajectory_length
    # pseudo_trajectory_length=args.pseudo_trajectory_length
else:
    pseudo_trajectory_length = trajectory_length

try_to_wrap_around_gradients = True
randomised_goal_position = False
randomised_state = True
refresh_unroll_frequency = 100
unroll_pseudo_initial_states_to_truth = True
initialise_wrap_around_gradients = True
#VISUALISATION
colors = ['red', 'blue', 'green', 'orange', 'black', 'yellow', 'purple', 'pink', 'olive', 'cyan']
plt.ion()
prinit = True
graphical = False
save = True
display = 5
print_time = 1
filename = str(args.trialname) + "_with_psi_restriction_" + str(with_psi_restriction) + "_randomised_state_" + str(
    randomised_state) + "_goal_" + str(goal) + "_test_" + testt + "_tanh_" + str(use_tanh)

def diff(t_a, t_b):
    t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

#BIKE PHYSICS
# Units in meters and kilograms
c = tf.constant(0.66)  # Horizontal distance between point where front wheel touches ground and centre of mass
d_cm =  tf.constant(0.30)  # Vertical distance between center of mass and cyclist
h =  tf.constant(0.94)  # Height of center of mass over the ground
l = tf.constant(1.11)  # Distance between front tire and back tire at the point where they touch the ground.
m_c =  tf.constant(15.0)  # mass of bicycle
m_d =  tf.constant(1.7)  # mass of tire
m_p =  tf.constant(60.0)  # mass of cyclist
r =  tf.constant(0.34)  # radius of tire
v =  tf.constant(10.0 / 3.6)  # velocity of the bicycle in m / s 2.7
goal_rsqrd =  tf.constant(1.0)
# Useful Precomputations
m = m_c + m_p
inertia_bc = (13. / 3) * m_c * h ** 2 + m_p * (h + d_cm) ** 2  # inertia of bicycle and cyclist
inertia_dv = (3. / 2) * (m_d * (r ** 2))  # Various inertia of tires
inertia_dl = .5 * (m_d * (r ** 2))  # Various inertia of tires
inertia_dc = m_d * (r ** 2)  # Various inertia of tires
sigma_dot = v / r
# Simulation constants
gravity =  tf.constant(9.82)
delta_time =  tf.constant(0.01)  # 0.01 # 0.054 m forward per delta time
# If omega exceeds +/- 12 degrees, the bicycle falls.
omega_range = tf.constant(np.array([[-np.pi * 12 / 180, np.pi * 12 / 180]] * batch_size))  # 12 degree in SI units.
theta_range = tf.constant(np.array([[-np.pi / 2, np.pi / 2]] * batch_size))
psi_range = tf.constant(np.array([[-np.pi, np.pi]] * batch_size))
rando = 0.0
yg = 60.
xg = 0.
goal_position = tf.constant(tf.random.uniform(minval=-50, maxval=50, shape=(batch_size, 2)) * (1 if randomised_goal_position else 0))
if not randomised_goal_position:
    goal_position += tf.constant([[xg, yg]] * batch_size)
def safe_divide(tensor_numerator, tensor_denominator):
    # attempt to avoid NaN bug in tf.where: https://github.com/tensorflow/tensorflow/issues/2540
    safe_denominator = tf.where(tf.not_equal(tensor_denominator, tf.zeros_like(tensor_denominator)),
                                tensor_denominator,
                                tensor_denominator + 1)
    return tensor_numerator / safe_denominator
def reset():
    # Lagoudakis (2002) randomizes the initial state "arout the equilibrium position"
    if randomised_state:
        theta = np.random.normal(0, 1, size=(batch_size, 1)) * np.pi / 180
        omega = np.random.normal(0, 1, size=(batch_size, 1)) * np.pi / 180
        thetad = np.zeros((batch_size, 1))
        omegad = np.zeros((batch_size, 1))
        omegadd = np.zeros((batch_size, 1))
        xb = np.random.uniform(-60, 60, (batch_size, 1))
        yb = np.zeros((batch_size, 1))
        xf = xb + (np.random.rand(batch_size, 1) * l - 0.5 * l) / 2  # halved it for psi
        yf = np.sqrt(l ** 2 - (xf - xb) ** 2) + yb
        psi = np.arctan((xb - xf) / (yf - yb))
        psig = psi - np.arctan(safe_divide((xb - xg), yg - yb))
        init_state = tf.Variable(tf.concat(
            [omega, omegad, omegadd, theta, thetad, xf, yf, xb, yb, psi, psig, np.zeros((batch_size, 1))],
            axis=1))
    else:
        theta = thetad = omega = omegad = omegadd = xf = yf = xb = yb = np.zeros((batch_size, 1))
        yf = yf + l
        psi = np.arctan((xb - xf) / (yf - yb))
        psig = psi - np.arctan(safe_divide((xb - xg), yg - yb))
        init_state = tf.Variable(tf.concat(
            [omega, omegad, omegadd, theta, thetad, xf, yf, xb, yb, psi, psig, np.zeros((batch_size, 1))],
            axis=1))
    # omega, omega_dot, omega_ddot, theta, theta_dot, x_f, y_f, x_b, y_b, psi,psig, timestep
    return init_state
# STATE initialisation
state_dimension = 12  # omega, omega_dot, omega_ddot, theta, theta_dot, x_f, y_f, x_b, y_b, psi,psig, timestep
initial_state = reset()
def step(state, action, trajectories_terminated, p_batch_size, corruption_to_physics_model=None):
    s = state
    omega = s[:, 0]
    omegad = s[:, 1]
    theta = s[:, 3]
    thetad = s[:, 4]  # theta - handle bar, omega - angle of bicycle to verticle psi = bikes angle to the yaxis
    xf = s[:, 5]
    yf = s[:, 6]
    xb = s[:, 7]
    yb = s[:, 8]
    psi = s[:, 9]
    timestep = s[:, -1]
    last_pos = s[:, 5:6]
    last_xf = xf
    last_yf = yf
    T = action[:, 0] * maximum_torque
    T = tf.where(T > maximum_torque, tf.ones_like(T) * maximum_torque, T)
    T = tf.where(T < -maximum_torque, tf.ones_like(T) * -maximum_torque, T)
    d = action[:, 1] * maximum_dis
    d = tf.where(d > maximum_dis, tf.ones_like(d) * maximum_dis, d)
    d = tf.where(d < -maximum_dis, tf.ones_like(d) * -maximum_dis, d)
    r_f = tf.where(theta == 0., tf.constant(1.e8), safe_divide(l, tf.abs(tf.sin(theta))))
    r_b = tf.where(theta == 0., tf.constant(1.e8), safe_divide(l, tf.abs(tf.tan(theta))))
    r_cm = tf.where(theta == 0., tf.constant(1.e8),
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
    # Must update omega based on PREVIOUS value of omegad.
    df = delta_time
    omegad += omegadd * df
    omega += omegad * df
    thetad += thetadd * df
    theta += thetad * df
    # Handlebars can't be turned more than 80 degrees.
    oneslikeTheta= tf.ones_like(theta)
    theta = tf.where(theta > 1.3963, oneslikeTheta * 1.3963, theta)
    theta = tf.where(theta < -1.3963, oneslikeTheta * -1.3963, theta)
    # Wheel ('tyre') contact positions.
    # ---------------------------------
    # Front wheel contact position.
    front_term = psi + theta + tf.sign(psi + theta) * tf.asin(v * df / (2. * r_f))
    back_term = psi + tf.sign(psi) * tf.asin(v * df / (2. * r_b))
    xf += v * df * -tf.sin(front_term)
    yf += v * df * tf.cos(front_term)
    xb += v * df * -tf.sin(back_term)
    yb += v * df * tf.cos(back_term)
    # Preventing numerical drift.
    # ---------------------------
    # Copying what Randlov did.
    current_wheelbase = tf.sqrt((xf - xb) ** 2 + (yf - yb) ** 2)
    relative_error = l / current_wheelbase - 1.0
    xb = tf.where(tf.abs(current_wheelbase - l) > 0.01, xb + (xb - xf) * relative_error, xb)
    yb = tf.where(tf.abs(current_wheelbase - l) > 0.01, yb + (yb - yf) * relative_error, yb)
    # Update heading, psi.
    # --------------------
    delta_y = yf - yb
    delta_goal_position = goal_position[:p_batch_size, :]
    delta_yg = delta_goal_position[:, 1] - yb
    psi = tf.where(tf.logical_and(xf == xb, delta_y < 0.0), pi,
                   tf.where((delta_y > 0.0),
                            tf.atan(safe_divide((xb - xf), delta_y)),
                            tf.sign(xb - xf) * 0.5 * pi - tf.atan(safe_divide(delta_y, (xb - xf)))))
    psig = tf.where(tf.logical_and(xf == xb, delta_yg < 0.0), psi - pi,
                    tf.where((delta_y > 0.0),
                             psi - tf.atan(safe_divide((xb - delta_goal_position[:, 0]), delta_yg)),
                             psi - tf.sign(xb - delta_goal_position[:, 0]) * 0.5 * pi - tf.atan(
                                 safe_divide(delta_yg, (xb - delta_goal_position[:, 0])))))
    omega = tf.reshape(omega, (p_batch_size, 1))
    omega = tf.where( tf.abs(omega) > pi / 2, tf.math.sign(omega) *pi / 2, omega)
    omega_dot = tf.reshape(omegad, (p_batch_size, 1))
    omega_ddot = tf.reshape(omegadd, (p_batch_size, 1))
    theta = tf.reshape(theta, (p_batch_size, 1))
    theta_dot = tf.reshape(thetad, (p_batch_size, 1))
    psig = tf.reshape(psig, (p_batch_size, 1))
    current_pos = tf.concat([tf.reshape(xf, [p_batch_size, 1]), tf.reshape(yf, [p_batch_size, 1])], axis=1)
    pos_d = current_pos - last_pos
    goal_displacement = delta_goal_position - current_pos
    goal_dist = tf.sqrt(tf.reduce_sum(tf.pow(goal_displacement, 2)))
    goal_displacement_normalised = safe_divide(goal_displacement, goal_dist)
    x_d = xf - last_xf
    y_d = yf - last_yf
    goal_displacement_x = delta_goal_position[:, 0] - xf
    goal_displacement_y = delta_goal_position[:, 1] - yf
    goal_dist = tf.sqrt(tf.pow(goal_displacement_x, 2) + tf.pow(goal_displacement_y, 2))
    goal_displacement_normalised_x = safe_divide(goal_displacement_x, goal_dist)  # constructing a unit vector here.
    goal_displacement_normalised_y = safe_divide(goal_displacement_y, goal_dist)
    if goal:
        r_t = tf.reduce_sum(pos_d * goal_displacement_normalised, axis=1)
        r_t = x_d * goal_displacement_normalised_x + y_d * goal_displacement_normalised_y  # this is a dot product
    else:
        y_d = yf - last_yf
        r_t = y_d
    timestep += 1.
    x_f = tf.reshape(xf, (p_batch_size, 1))
    y_f = tf.reshape(yf, (p_batch_size, 1))
    x_b = tf.reshape(xb, (p_batch_size, 1))
    y_b = tf.reshape(yb, (p_batch_size, 1))
    psi = tf.reshape(psi, (p_batch_size, 1))
    r_t = tf.reshape(r_t, (p_batch_size, 1))
    timestep = tf.reshape(timestep, (p_batch_size, 1))
    trajectories_terminating = timestep >= pseudo_trajectory_length
    trajectories_terminating = tf.reshape(trajectories_terminating, [p_batch_size,])
    new_state = tf.concat([omega, omega_dot, omega_ddot, theta, theta_dot, x_f, y_f, x_b, y_b, psi, psig, timestep],
                          axis=1)
    penalty_handle = flat_bottomed_barrier_function(tf.abs(theta), 1.3963 * 0.9, 8)
    penalty_angle = flat_bottomed_barrier_function(tf.abs(omega), pi / 15, 8)
    penalty_psi = 0
    penalty_psig = 0
    penalty_psi = flat_bottomed_barrier_function(tf.abs(psig), pi / 2, 8) * (1 if with_psi_restriction else 0)
    psiRemoved = int(not(testt == "psiRemoved"))
    angleRemoved = int(not(testt == "angleRemoved"))
    handleRemoved = int(not(testt == "handleRemoved"))
    recorded_tanh = -tf.tanh(penalty_psi * psiRemoved + penalty_angle*angleRemoved + penalty_handle*handleRemoved) + r_t
    if use_tanh:
        reward = -tf.tanh(penalty_psi * psiRemoved + penalty_angle*angleRemoved + penalty_handle*handleRemoved) + r_t
    else:
        reward = -1 * (penalty_psi * psiRemoved + penalty_angle*angleRemoved + penalty_handle*handleRemoved) + r_t
    return [reward, recorded_tanh, new_state, trajectories_terminating]
def flat_bottomed_barrier_function(x, k_width, k_power):
    return tf.pow(tf.maximum(x / (k_width * 0.5) - 1, 0), k_power)
def evaluate_final_state(state):
    return tf.zeros_like(state[:, 0])
#MODEL DESIGN
learning_rate = 0.01
unroll_pseudo_initial_states_to_truth = True
initialise_wrap_around_gradients = True
class model(keras.Model):
    def __init__(self):
        super(model, self).__init__()
        self.neural_layers = []
        for hidden in num_hidden_units:
            self.neural_layers.append(keras.layers.Dense(hidden, activation="tanh",
                                                         kernel_initializer=keras.initializers.RandomNormal(
                                                             stddev=0.001),
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
def converter(state, passed_batch_size):
    # omega, omega_dot, omega_ddot, theta, theta_dot, x_f, y_f, x_b, y_b, psi,psig, timestep
    omega = state[:, 0]
    omega_dot = state[:, 1]
    theta = state[:, 3]
    theta_dot = state[:, 4]  # theta - handle bar, omega - angle of bicycle to verticle
    psi = state[:, 9]
    psig = state[:, 10]
    psi = tf.reshape(psi, (passed_batch_size, 1))
    omega = tf.reshape(omega, (passed_batch_size, 1))
    omega_dot = tf.reshape(omega_dot, (passed_batch_size, 1))
    theta = tf.reshape(theta, (passed_batch_size, 1))
    theta_dot = tf.reshape(theta_dot, (passed_batch_size, 1))
    psig = tf.reshape(psig, (passed_batch_size, 1))
    omega_visible = tf.tanh(omega * 10)
    omega_dot = tf.tanh(omega_dot)
    theta_dot = tf.tanh(theta_dot)
    theta = tf.tanh(theta / (pi / 4))
    if goal:
        converted_state = tf.concat([omega_visible, omega_dot, theta, theta_dot, tf.sin(psig), tf.cos(psig)], axis=1)
    else:
        converted_state = tf.concat([omega_visible, omega_dot, theta, theta_dot, tf.sin(psi), tf.cos(psi)], axis=1)
    return converted_state
def expand_trajectories(start_states, final_artificial_gradient, p_batch_size):
    total_rewards = tf.constant(0.0, shape=[p_batch_size])
    recorded_total_rewards = tf.constant(0.0, shape= [p_batch_size])
    actions = tf.zeros((p_batch_size, action_space))
    trajectories_terminated = tf.cast(tf.zeros_like(start_states[:, 0]), tf.bool)
    trajectories_terminated_list = []
    state = start_states
    action_list = []
    trajectory_list = [state]
    # build main graph.  This is a long graph with unrolled in time for trajectory_length steps.  Each step includes one neural network followed by one physics-model
    for t in range(trajectory_length):
        converted_state = converter(state, p_batch_size)
        prevaction = keras_action_network(converted_state)
        action = tf.reshape(prevaction, (p_batch_size, action_space))
        [rewards,recorded_tanh, n_state, trajectories_terminating] = step(state, action, trajectories_terminated, p_batch_size)
        state = tf.where(tf.expand_dims(trajectories_terminated, 1), state, n_state)
        action_list.append(prevaction)
        rewards = tf.reshape(rewards, (p_batch_size,))
        recorded_tanh = tf.reshape(recorded_tanh, (p_batch_size,))

        if t == trajectory_length - 1:
            # this is the final step of this trajectory chunk.  If this trajectory chunk feeds into the next trajectory chunk, then feed the gradients through too.
            correction = tf.reduce_sum((n_state - tf.stop_gradient(n_state)) * final_artificial_gradient,
                                       axis=1)  # This adds in the gradient that was passed in.  This gradient will have come out of the START of the next trajectory chunk, so it gets added into the END of this current trajectory.
            rewards += correction
        total_rewards += tf.where(trajectories_terminated, tf.zeros_like(rewards), rewards)
        total_rewards += tf.where(tf.logical_and(trajectories_terminating, tf.logical_not(trajectories_terminated)),
                                  evaluate_final_state(state), tf.zeros_like(rewards))
        recorded_total_rewards += tf.where(trajectories_terminated, tf.zeros_like(recorded_tanh), recorded_tanh)
        recorded_total_rewards += tf.where(tf.logical_and(trajectories_terminating, tf.logical_not(trajectories_terminated)),
                                  evaluate_final_state(state), tf.zeros_like(recorded_tanh))

        trajectories_terminated = tf.logical_or(trajectories_terminated, trajectories_terminating)
        trajectory_list.append(state)
    action_history = tf.stack(action_list, axis=0)
    trajectory = tf.stack(trajectory_list, axis=0)
    average_total_reward = tf.reduce_mean(total_rewards)
    average_total_recorded_reward = tf.reduce_mean(recorded_total_rewards)
    return [average_total_reward,average_total_recorded_reward, trajectory, action_history, trajectories_terminated]
opt = keras.optimizers.Adam(learning_rate)

@tf.function
def dolearn(iteration,start_states, final_artificial_gradient):
    if (iteration == 0 or (refresh_unroll_frequency > 0 and iteration % refresh_unroll_frequency == 0)):
        if unroll_pseudo_initial_states_to_truth and pseudo_trajectory_length > trajectory_length:
            [total_reward, trajectory, action_hisotry, trajectories_terminated] = expand_trajectories(start_states,
                                                                                                      final_artificial_gradient,
                                                                                                      pseudo_batch_size)
            #trajectory = tf.reshape(trajectory[:, 1:, :],[pseudo_batch_size, pseudo_trajectory_length // trajectory_length,trajectory_length, -1])
            trajectory_final_states = trajectory[:, 1:, :]
            trajectory_final_states = trajectory_final_states[-1, :, :]
            #trajectory_final_states = tf.reshape(trajectory_final_states, [batch_size-1, -1])
            #start_states[pseudo_batch_size:, 0:2] = trajectory_final_states[:-pseudo_batch_size, 0:2]
            start_state_sliced = start_states[:pseudo_batch_size + 1, :]
            trajectory_final_states_sliced = trajectory_final_states[:-pseudo_batch_size, :]
            start_states = tf.concat(
                [start_state_sliced, trajectory_final_states_sliced], axis=0)
        if initialise_wrap_around_gradients and try_to_wrap_around_gradients and pseudo_trajectory_length > trajectory_length:
            gradients_list = []
            d_reward_list = []
            gradients = tf.zeros_like(start_states[:pseudo_batch_size])
            for i in reversed(range(pseudo_trajectory_length // trajectory_length)):
                initial_states = start_states[i * pseudo_batch_size:(i + 1) * pseudo_batch_size]
                with tf.GradientTape() as tape:
                    tape.watch(initial_states)
                    [total_reward, trajectory, action_hisotry, trajectories_terminated] = expand_trajectories(
                        initial_states,
                        final_artificial_gradient[
                        i * pseudo_batch_size:(
                                                      i + 1) * pseudo_batch_size,
                        :], pseudo_batch_size)
                    loss = -total_reward / pseudo_batch_size
                gradients = tape.gradient(loss, initial_states)
                gradients_list.append(gradients[1:])
                d_reward_list.append(-gradients[0])
            gradients_list.reverse()
            d_reward_list.reverse()
            gradients_list = tf.concat(gradients_list, axis=0)
            d_reward_list = tf.concat(d_reward_list, axis =0)
    return gradients_list, d_reward_list, trajectory, total_reward, action_hisotry, trajectories_terminated

@tf.function
def dolearn2(iteration,start_states, final_artificial_gradient):
    with tf.GradientTape() as t:
        t.watch(start_states)
        [total_reward,recorded_total_reward, trajectory, action_hisotry,trajectories_terminated] = expand_trajectories(start_states,final_artificial_gradient, batch_size)
        cost_ = -tf.reduce_mean(total_reward)
    grads = t.gradient(cost_, [start_states]+keras_action_network.trainable_weights)
    dCost_dWeights = grads[1:]
    dReward_dInputState = -grads[0]
    return dCost_dWeights,dReward_dInputState, trajectory, total_reward,recorded_total_reward,action_hisotry,trajectories_terminated

#GRAPHICS
def static_graphics():
    fig, ((ax_omega, ax_theta), (ax_trajectory, ax_reward_history), (ax_actionT, ax_psi),
          (ax_actiond, ax_timestep)) = plt.subplots(nrows=4, ncols=2, figsize=(10, 10))
    # display.clear_output(wait=True)
    # fig=plt.figure(figsize=[12.4, 4.8])
    fig.tight_layout(pad=5.0)
    # ax_omega = omega
    pad = 10
    ax_omega.axis(
        [0, trajectory_length, (-math.pi / 15) * 180 / math.pi - pad, (math.pi / 15) * 180 / math.pi + pad])
    ax_omega.set(xlabel='timestep', ylabel='Bike Roll value in Degrees')
    ax_omega.set_title('(Bike Roll).')
    ax_omega.grid()
    # ax2 = theta
    ax_theta.axis([0, trajectory_length, -80 - pad, 80 + pad])
    ax_theta.set(xlabel='timestep', ylabel='Bike handle value in Degrees')
    ax_theta.set_title('(Bike Handle).')
    ax_theta.grid()
    # ax3 = Agent moving in the field
    ax_trajectory.axis([-b, b, -b, b])
    ax_trajectory.set(xlabel='x', ylabel='y')
    ax_trajectory.set_title('Bike Trajectory.')
    ax_trajectory.plot(goal_position[:, 0], goal_position[:, 1], color='green', marker='o')
    ax_trajectory.grid()
    # ax4 = reward over time.
    ax_reward_history.axis([0, max_iterations, -1 - 0.5, 1 + 0.5])
    ax_reward_history.set(xlabel='Iteration', ylabel='Reward')
    ax_reward_history.set_title('Reward over Iteration')
    for traj in range(batch_size):
        trajectory_x_coord = [0]  # trajectory[:,traj,0]
        ax_reward_history.plot(trajectory_x_coord)
    ax_reward_history.grid()
    # ax5 = action the agent takes
    ax_actionT.axis([0, trajectory_length, -2 - 0.5, 2 + 0.5])
    ax_actionT.legend(loc="upper right")
    ax_actionT.set(xlabel='timestep', ylabel='torque')
    ax_actionT.set_title('Trajectory torque')
    ax_actionT.grid()
    # ax6 = agent psi
    ax_psi.axis([0, trajectory_length, -180 - pad, 180 + pad])
    ax_psi.set(xlabel='timestep', ylabel='psi')
    ax_psi.set_title('Bike direction Psi')
    ax_psi.grid()
    # ax7 = agent d
    ax_actiond.axis([0, trajectory_length, -maximum_dis - 0.01, maximum_dis + 0.01])
    ax_actiond.set(xlabel='timestep', ylabel='displacement')
    ax_actiond.set_title('The centre of mass displacement')
    ax_actiond.grid()
    # ax8 timestep
    ax_timestep.axis([0, max_iterations, 0 - pad, trajectory_length + pad])
    ax_timestep.set_title('max balancing duration')
    ax_timestep.set(xlabel='iteration', ylabel='time steps')
    ax_timestep.grid()
    plt.draw()
    plt.pause(0.001)
    return [fig, ax_omega, ax_theta, ax_trajectory, ax_reward_history, ax_actionT, ax_psi, ax_actiond, ax_timestep]
def dynamic_graphics(trajectory, stats, passed_actions,reward_history, timestep_history):
    # omega, omega_dot, omega_ddot, theta, theta_dot, x_f, y_f, x_b, y_b, psi,psig, timestep
    fig, ax_omega, ax_theta, ax_trajectory, ax_reward_history, ax_actionT, ax_psi, ax_actiond, ax_timestep = stats
    T = passed_actions[:, :, 0] * maximum_torque
    T = tf.where(T > maximum_torque, tf.ones_like(T) * maximum_torque, T)
    T = tf.where(T < -maximum_torque, tf.ones_like(T) * -maximum_torque, T)
    d = passed_actions[:, :, 1] * maximum_dis
    d = tf.where(d > maximum_dis, tf.ones_like(d) * maximum_dis, d)
    d = tf.where(d < -maximum_dis, tf.ones_like(d) * -maximum_dis, d)
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
    col = 0
    ax_omega.lines.clear()
    ax_theta.lines.clear()
    ax_trajectory.lines.clear()
    ax_reward_history.lines.clear()
    ax_actionT.lines.clear()
    ax_psi.lines.clear()
    ax_actiond.lines.clear()
    ax_timestep.lines.clear()
    for trajectory_number in range(batch_size):
        if trajectory[0, trajectory_number, -1] == 0:
            col = (col + 1) % len(colors)
        conv_omega = omega[:, trajectory_number] * 180 / np.pi
        conv_theta = theta[:, trajectory_number] * 180 / np.pi
        conv_psi = psi[:, trajectory_number] * 180 / np.pi
        ax_omega.plot(time_steps[:, trajectory_number], conv_omega, color=colors[col])
        ax_theta.plot(time_steps[:, trajectory_number], conv_theta, color=colors[col])
        x = x_f[:, trajectory_number]
        y = y_f[:, trajectory_number]
        ax_trajectory.plot(x, y, color=colors[col])
        ax_actionT.plot(time_steps[1:, trajectory_number], T[:, trajectory_number], color=colors[col])
        ax_actiond.plot(time_steps[1:, trajectory_number], d[:, trajectory_number], color=colors[col])
        ax_psi.plot(time_steps[:, trajectory_number], conv_psi, color=colors[col])
    ax_trajectory.plot(goal_position[0, 0], goal_position[0, 1], color='b', marker='o')
    ax_reward_history.axis([0, max_iterations, min(reward_history) - 10, max(reward_history) + 10])
    ax_reward_history.plot(reward_history, color='red')
    ax_timestep.plot(timestep_history, color='red')
    plt.draw()
    plt.pause(0.001)
if graphical:
    stat = static_graphics()

#TRAINING
timestep_history = []
action_history = []
reward_history = []
trajectory_history = []
initial_state_backup = tf.identity(initial_state)
trajectories_terminated = tf.cast(tf.zeros_like(initial_state[:, 0]), tf.bool)
t_a = datetime.now()
t_b = datetime.now()
final_artificial_gradient = np.zeros_like(initial_state)
print(batch_size)
for iteration in range(max_iterations):
    state1 = tf.identity(initial_state)
    # trajectories_terminated = trajectories_terminated
    gradients_list,d_reward_list, trajectory, total_reward,recorded_total_reward, actions, trajectories_terminated = dolearn2(iteration,state1,final_artificial_gradient)
    trajectory = trajectory.numpy()
    trajectories_terminated = trajectories_terminated.numpy()
    for _ in gradients_list:
        if np.isnan(_).any():
            print("Nan Grads")

    for attempts in range(pseudo_trajectory_length // trajectory_length):
        for i_ in range(1, batch_size):
            if trajectories_terminated[i_ - 1] or i_ %10 ==0:
                # the previous trajectory crashed, so the next trajectory needs to start from the beginning
                # do trajectory wrap-around hack (feed-foward wrap-around of start states)
                initial_state[i_, :].assign(initial_state_backup[i_, :])
                # do trajectory wrap-around hack (backwards wrap-around of gradients)
            else:
                # the previous trajectory did not crash, so it is still going! So let the next trajectory start from where the old one left off...
                # (This is to reposition the "time-step" dimension into the batch-size dimension.  We do this because we'll get much better parallelism
                # on the graphics card / inner c++ loops if we do this.  However it's only approximate - there might be tiny gaps appearing in a set of trajectories
                # which are linked together like this)
                # do trajectory wrap-around hack (feed-foward wrap-around of start states)
                initial_state[i_, :].assign( trajectory[-1, i_ - 1,
                                       :])  # copy over every state variable so that next trajectory starts where old one ended.
    opt.apply_gradients(zip(gradients_list, keras_action_network.trainable_weights))
    print(trajectory[:,:,-1])
    # total_reward = total_reward + (-1 *(trajectory_length - trajectory[-1,:,-1]))
    average_total_reward_stepwise = np.max(recorded_total_reward.numpy())
    reward_history.append(average_total_reward_stepwise)
    timestep_history.append(np.max(trajectory[:, :,-1]))
    action_history.append(actions)
    trajectory_history.append(trajectory)
    if iteration % print_time == 0:
        if graphical:
            dynamic_graphics(trajectory, stat, actions)
        if prinit:
            t_b = datetime.now()
            dt = t_b - t_a
            print("iteration: ", iteration, "// Average_total_reward_step_wise: ", average_total_reward_stepwise,
                  "Average Step: ", np.mean(trajectory[-1, :, -1]), "in steps and ",
                  np.mean(trajectory[-1, :, -1]) * delta_time, "in seconds", "time taken from last iter: ",
                  diff(t_a, t_b))
            t_a = t_b

    if save:
        if iteration % 10 == 0:
            save_mat = np.concatenate([[reward_history], [timestep_history]], axis=0)
            np.save("runs/" + trial_name + "_marker_" + str(iteration) + "_results_" + filename + ".npy",save_mat,allow_pickle=True)
            np.save("runs/" + trial_name + "_marker_" + str(iteration) + "_action_history_" + filename + ".npy",np.array(action_history),allow_pickle=True)
            np.save("runs/" + trial_name + "_marker_" + str(iteration) + "_trajectory_history_" + filename + ".npy",np.array(trajectory_history),allow_pickle=True)
            np.save("runs/" + filename + ".npy", trajectory,allow_pickle=True)
            action_history = []
            reward_history = []
            timestep_history = []
            trajectory_history = []
        if iteration % 10 == 0:
            keras_action_network.save_weights("./checkpoints/my_checkpoint")
if save:
    np.save("runs/" + trial_name + "_marker_" + str(iteration) + "_results_" + filename + ".npy",save_mat,allow_pickle=True)
    np.save("runs/last_state_" + filename + ".npy", trajectory,allow_pickle=True)
    keras_action_network.save_weights("./checkpoints/my_checkpoint")