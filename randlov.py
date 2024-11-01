import numpy as np
from numpy import sin, cos, tan, sqrt, arcsin, arctan, sign
import tensorflow as tf
import math
batch_size = 1
randomised_state = True
def reset():
    # Lagoudakis (2002) randomizes the initial state "arout the
    # equilibrium position"
    yg = 100.
    xg = 0.
    l = tf.constant(1.11,
                    tf.float64)  # Distance between front tire and back tire at the point where they touch the ground.
    if randomised_state:
        theta = np.random.normal(0, 1,size= (batch_size,1)) * np.pi / 180
        omega = np.random.normal(0, 1,size=(batch_size,1)) * np.pi / 180
        thetad = omegad = omegadd = xb = yb = np.zeros((batch_size,1))
        xf = xb + (np.random.rand(batch_size,1) * l - 0.5 * l)
        yf = np.sqrt(l ** 2 - (xf - xb) ** 2) + yb
        psi = np.arctan((xb - xf) / (yf - yb))
        psig = psi - np.arctan(safe_divide((xb - xg), yg - yb))
        init_state = tf.cast(np.concatenate([omega, omegad, omegadd, theta, thetad, xf, yf, xb, yb, psi, psig, np.zeros((batch_size,1))], axis=1), tf.float64)

    else:
        theta = thetad = omega = omegad = omegadd = xf = yf = xb = yb = np.zeros((batch_size,1))
        yf = yf +l
        psi = np.arctan((xb - xf) / (yf - yb))
        psig = psi - np.arctan(safe_divide((xb - xg), yg - yb))
        init_state = tf.cast(np.concatenate(
            [omega, omegad, omegadd, theta, thetad, xf, yf, xb, yb, psi, psig, np.zeros((batch_size, 1))], axis=1),
                tf.float64)

    # omega, omega_dot, omega_ddot, theta, theta_dot, x_f, y_f, x_b, y_b, psi,psig, timestep
    return init_state
def safe_divide(tensor_numerator, tensor_denominator):
    # attempt to avoid NaN bug in tf.where: https://github.com/tensorflow/tensorflow/issues/2540
    safe_denominator = tf.where(tf.not_equal(tensor_denominator, tf.zeros_like(tensor_denominator,tf.float64)), tensor_denominator,
                                tensor_denominator + 1)
    return tensor_numerator / safe_denominator
def randlov_step(state,actions):
    # Acceleration on Earth's surface due to gravity (m/s^2):
    g = 9.82
    # See the paper for a description of these quantities:
    # Distances (in meters):
    c = 0.66
    dCM = 0.30
    h = 0.94
    L = 1.11
    r = 0.34
    # Masses (in kilograms):
    Mc = 15.0
    Md = 1.7
    Mp = 60.0
    # Velocity of a bicycle (in meters per second), equal to 10 km/h:
    v = 10.0 * 1000.0 / 3600.0
    # Derived constants.
    M = Mc + Mp  # See Randlov's code.
    Idc = Md * r ** 2
    Idv = 1.5 * Md * r ** 2
    Idl = 0.5 * Md * r ** 2
    Itot = 13.0 / 3.0 * Mc * h ** 2 + Mp * (h + dCM) ** 2
    sigmad = v / r
    s = state
    omega = s[ 0]
    omegad = s[ 1]
    theta = s[ 3]
    thetad = s[ 4]  # theta - handle bar, omega - angle of bicycle to verticle psi = bikes angle to the yaxis
    xf = s[ 5]
    yf = s[ 6]
    xb = s[ 7]
    yb = s[ 8]
    psi = s[ 9]
    # psi = tf.zeros_like(psi,tf.float64)
    psig = s[10]
    time_step = s[ 11]
    delta_time = 0.02
    (T, d) = actions

    # store a last states
    last_xf = xf
    last_yf = yf
    last_omega = omega
    last_psig = psig
    # Intermediate time-dependent quantities.
    # ---------------------------------------
    # Avoid divide-by-zero, just as Randlov did.
    if theta == 0:
        rf = 1e8
        rb = 1e8
        rCM = 1e8
    else:
        rf = L / np.abs(sin(theta))
        rb = L / np.abs(tan(theta))
        rCM = sqrt((L - c) ** 2 + L ** 2 / tan(theta) ** 2)

    phi = omega + np.arctan(d / h)

    # Equations of motion.
    # --------------------
    # Second derivative of angular acceleration:
    omegadd = 1 / Itot * (M * h * g * sin(phi)
                               - cos(phi) * (Idc * sigmad * thetad
                                             + sign(theta) * v ** 2 * (
                                                     Md * r * (1.0 / rf + 1.0 / rb)
                                                     + M * h / rCM)))
    thetadd = (T - Idv * sigmad * omegad) / Idl

    # Integrate equations of motion using Euler's method.
    # ---------------------------------------------------
    # yt+1 = yt + yd * dt.
    # Must update omega based on PREVIOUS value of omegad.
    omegad += omegadd * delta_time
    omega += omegad * delta_time
    thetad += thetadd * delta_time
    theta += thetad * delta_time

    # Handlebars can't be turned more than 80 degrees.
    theta = np.clip(theta, -1.3963, 1.3963)

    # Wheel ('tyre') contact positions.
    # ---------------------------------

    # Front wheel contact position.
    front_temp = v * delta_time / (2 * rf)
    # See Randlov's code.
    if front_temp > 1:
        front_temp = sign(psi + theta) * 0.5 * np.pi
    else:
        front_temp = sign(psi + theta) * arcsin(front_temp)
    xf += v * delta_time * -sin(psi + theta + front_temp)
    yf += v * delta_time * cos(psi + theta + front_temp)

    # Rear wheel.
    back_temp = v * delta_time / (2 * rb)
    # See Randlov's code.
    if back_temp > 1:
        back_temp = np.sign(psi) * 0.5 * np.pi
    else:
        back_temp = np.sign(psi) * np.arcsin(back_temp)
    xb += v * delta_time * -sin(psi + back_temp)
    yb += v * delta_time * cos(psi + back_temp)

    # Preventing numerical drift.
    # ---------------------------
    # Copying what Randlov did.
    current_wheelbase = sqrt((xf - xb) ** 2 + (yf - yb) ** 2)
    if np.abs(current_wheelbase - L) > 0.01:
        relative_error = L / current_wheelbase - 1.0
        xb += (xb - xf) * relative_error
        yb += (yb - yf) * relative_error

    # Update heading, psi.
    # --------------------
    delta_y = yf - yb
    if (xf == xb) and delta_y < 0.0:
        psi = np.pi
    else:
        if delta_y > 0.0:
            psi = arctan((xb - xf) / delta_y)
        else:
            # TODO we inserted this ourselves:
            # delta_x = xb - xf
            # if delta_x == 0:
            #    dy_by_dx = np.sign(delta_y) * np.inf
            # else:
            #    dy_by_dx = delta_y / delta_x
            psi = sign(xb - xf) * 0.5 * np.pi - arctan(delta_y / (xb - xf))
        # dy_by_dx))
    time_step +=1
    sensors = np.array([omega, omegad, omegadd, theta, thetad,
                             xf, yf, xb, yb, psi, psig,time_step])

    return sensors

def flat_bottomed_barrier_function(x, k_width, k_power):
    return tf.pow(tf.maximum(x / (k_width * 0.5) - 1, 0), k_power)

def step(state, action, trajectories_terminated, corruption_to_physics_model=None):
    display = 8
    action_is_theta = True
    maximum_dis = 0.02  # 0.02
    maximum_torque = 2.
    batch_size = 1
    action_space = 2 if action_is_theta else 1
    num_hidden_units = [16, 16]
    trajectory_length = 3000
    max_iterations = 1000
    print_time = 1
    ## BIKE STATS
    # Units in meters and kilograms
    c = 0.66  # Horizontal distance between point where front wheel touches ground and centre of mass
    d_cm = 0.30  # Vertical distance between center of mass and cyclist
    h = 0.94  # Height of center of mass over the ground
    l = tf.constant(1.11,
                    tf.float64)  # Distance between front tire and back tire at the point where they touch the ground.
    m_c = 15.0  # mass of bicycle
    m_d = 1.7  # mass of tire
    m_p = 60.0  # mass of cyclist
    r = 0.34  # radius of tire
    v = 10.0 * 1000.0 / 3600.0  # velocity of the bicycle in m / s 2.7
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
    delta_time = 0.02  # 0.054 m forward per delta time
    sim_steps = 1
    randomised_goal_position = False
    randomised_state = False
    randomised_position = False
    yg = 100.
    xg = 0.
    goal_position = tf.cast(
        (np.random.rand(batch_size, 2)) * ([[1., 1.]] if randomised_position else [[0., 0.]] * batch_size) + [
            [xg, yg]] * batch_size,
        tf.float64)
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
    psig = s[:, 10]
    timestep = s[:, 11]
    # store a last states
    last_xf = xf
    last_yf = yf
    last_omega = omega
    last_psig = psig
    T = action[:, 0]
    T = tf.where(T > maximum_torque, tf.constant(tf.ones_like(T) * maximum_torque, tf.float64), T)
    T = tf.where(T < -maximum_torque, tf.constant(tf.ones_like(T) * -maximum_torque, tf.float64), T)
    d = action[:, 1]
    d = tf.where(d > maximum_dis, tf.constant(tf.ones_like(d) * maximum_dis, tf.float64), d)
    d = tf.where(d < -maximum_dis, tf.constant(tf.ones_like(d) * -maximum_dis, tf.float64), d)
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
    yf += v*df * tf.cos(front_term)
    xb += v*df * -tf.sin(back_term)
    yb += v*df * tf.cos(back_term)

    '''
    front_temp = v * df / (2 * r_f)
    # See Randlov's code.
    front_temp = tf.where(front_temp > 1, tf.sign(psi + theta) * 0.5 * math.pi, tf.sign(psi + theta) * tf.asin(front_temp))
    xf = xf + (v *df * (-tf.sin(psi + theta + front_temp)))
    yf = yf + (v *df * tf.cos(psi + theta + front_temp))
    # Rear wheel.
    back_temp = v * df / (2 * r_b)
    # See Randlov's code.
    back_temp = tf.where(back_temp>1,tf.sign(psi) * 0.5 * math.pi, tf.sign(psi) * tf.asin(back_temp))
    xb =xb+ (v *df* (-tf.sin(psi + back_temp)))
    yb =yb+ (v  *df* tf.cos(psi + back_temp))
    '''
    '''
    front_temp = psi + theta + tf.sign(psi + theta) * tf.asin(tf.tanh(v * delta_time / (2 * r_f)))
    xf += -tf.sin(front_temp)
    yf += tf.cos(front_temp)
    back_temp = psi + tf.sign(psi) * tf.asin(tf.tanh(v * delta_time / (2. * r_b)))
    xb += -tf.sin(back_temp)
    yb += tf.cos(back_temp)
    '''
    # Handle Roundoff errors, to keep the length of the bicycle constant
    '''
    dist = tf.sqrt((xf - xb) ** 2 + (yf - yb) ** 2)
    xb += tf.where(tf.abs(dist - l) > 0.01, (xb - xf) * (l - dist) / dist, 0)
    yb += tf.where(tf.abs(dist - l) > 0.01, (yb - yf) * ((l - dist) / dist), 0)
    '''
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
    old_psi =psi
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
    # if (fabs(omega) > (pi/15)) { /* the bike has fallen over */
    r_t = yf - last_yf
    # new_state = np.array([omega, omega_dot, omega_ddot, theta, theta_dot])
    x_f = tf.reshape(xf, (batch_size, 1))
    y_f = tf.reshape(yf, (batch_size, 1))
    x_b = tf.reshape(xb, (batch_size, 1))
    y_b = tf.reshape(yb, (batch_size, 1))
    psi = tf.reshape(psi, (batch_size, 1))
    r_t = tf.reshape(r_t, (batch_size, 1))
    timestep = tf.reshape(timestep, (batch_size, 1))
    timestep += 1.

    trajectories_terminating = tf.logical_or(timestep >= trajectory_length, tf.abs(omega) > math.pi / 15)
    trajectories_terminating = tf.logical_or(trajectories_terminating, tf.abs(psi)> math.pi/2)
    # trajectories_terminating = tf.logical_or(timestep >= trajectory_length, tf.abs(omega) > math.pi / 15)
    trajectories_terminating = tf.reshape(trajectories_terminating, [batch_size, ])

    timestep = tf.reshape(timestep, (batch_size, 1))
    # omega, omega_dot, omega_ddot, theta, theta_dot, x_f, y_f, x_b, y_b, psi,psig, timestep
    new_state = tf.concat([omega, omega_dot, omega_ddot, theta, theta_dot, x_f, y_f, x_b, y_b, psi, psig, timestep],
                          axis=1)
    penalty_handle = flat_bottomed_barrier_function(tf.abs(theta), 1.3963 * 0.9, 8)
    penalty_angle = flat_bottomed_barrier_function(tf.abs(omega), math.pi / 30, 8)
    penalty_psi = flat_bottomed_barrier_function(tf.abs(psi), math.pi / 4, 8)
    # xy = tf.concat([x_f, y_f], axis=1)

    reward = -tf.tanh(penalty_angle + penalty_handle + penalty_psi) + r_t

    return [reward, new_state, trajectories_terminating]

column = ["omega", "omegad", "omegadd", "theta", "thetad","xf", "yf", "xb", "yb", "psi", "psig","time_step"]
initial_state = reset()
initial_state2 = initial_state.numpy()[0]
action = [np.random.uniform(-2,2),np.random.uniform(-0.02,0.02)]
action_ = tf.constant([[action[0],action[1]]*batch_size],tf.float32)
print("our engine  //  Randlov engine")
for i in range(100):

    _, a,_ = step(initial_state,action_,tf.zeros_like(initial_state[0]))
    b = randlov_step(initial_state2,action)
    initial_state = a
    initial_state2 = b
    action = [np.random.uniform(-2,2),np.random.uniform(-0.02,0.02)]
    action_ = tf.constant([[action[0],action[1]]*batch_size],tf.float32)
    for i in range(12):
        print(column[i],"//",a[0][i].numpy()," vs ",b[i], "IS checked= ", (a[0][i] == b[i]).numpy())


