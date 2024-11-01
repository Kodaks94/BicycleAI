import numpy as np
import tensorflow as tf

def calculate_front_wheel_position_numpy(psi, theta, v, df, rf):
    front_temp = v * df / (2 * rf)
    if front_temp > 1:
        front_temp = np.sign(psi + theta) * 0.5 * np.pi
    else:
        front_temp = np.sign(psi + theta) * np.arcsin(front_temp)
    xf = v * df * -np.sin(psi + theta + front_temp)
    yf = v * df * np.cos(psi + theta + front_temp)
    return xf, yf

def calculate_front_wheel_position_tensorflow(psi, theta, v, df, rf):
    front_term = psi + theta + tf.sign(psi + theta) * tf.asin(v * df / (2. * rf))
    xf = v * df * -tf.sin(front_term)
    yf = v * df * tf.cos(front_term)
    return xf, yf

# Test inputs
psi = 0.1
theta = 0.2
v = 2.0
df = 0.05
rf = 0.3

# Calculate positions using numpy
xf_numpy, yf_numpy = calculate_front_wheel_position_numpy(psi, theta, v, df, rf)

# Calculate positions using TensorFlow
xf_tensorflow, yf_tensorflow = calculate_front_wheel_position_tensorflow(psi, theta, v, df, rf)

# Print the results
print("Numpy implementation:")
print("xf:", xf_numpy)
print("yf:", yf_numpy)

print("\nTensorFlow implementation:")
print("xf:", xf_tensorflow.numpy())
print("yf:", yf_tensorflow.numpy())