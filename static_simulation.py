import numpy as np
import random
import matplotlib.pyplot as plt
from KalmanFilter import KalmanFilter

random.seed(42)
np.random.seed(42)


# -------------------- Tests: Kalman Filter --------------------  
num_samples = 100
add_noise_flag = True

# -----
#   Create dummy position data for simulation:
locations = np.expand_dims(np.hstack((np.expand_dims(np.arange(0, num_samples), 1), np.expand_dims(np.arange(0, num_samples), 1))), axis=2)
measurements = locations + (np.random.randn(num_samples, 2, 1) if add_noise_flag else 0)
# ---

# -----
#   Initialize a Kalman Filter for 2D (object) tracking with no control inputs, 
#   internal state representation of [pos_x, pos_y, velocity_x, veloctiy_y], and measurements of [pos_x, pos_y].

dt = 1 # duration of discrete time steps
var_acceleration_x = 1 # variance of the acceleration in the x direction
var_acceleration_y = 1 # variance of the acceleration in the y direction

A = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0 , 1]
]) # state transition matrix of the process from state at time k to the state at time k+1 (sometimes denoted as Phi)

P = np.array([
    [1000, 0, 0, 0],
    [0, 1000, 0, 0],
    [0, 0, 1000, 0],
    [0, 0, 0, 1000]
]) # error covariance matrix

H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
]) # (measurement matrix) noiseless connection btw the state vector and the measurement vector

Q = np.array([
    [(dt**4)*(var_acceleration_x**2)/4, 0, 0, (dt**3)*(var_acceleration_x**2)/2],
    [0, (dt**4)*(var_acceleration_y**2)/4, 0, (dt**3)*(var_acceleration_y**2)/2],
    [(dt**3)*(var_acceleration_x**2)/2, 0, (dt**2)*(var_acceleration_x**2), 0],
    [0, (dt**3)*(var_acceleration_y**2)/2, 0, (dt**2)*(var_acceleration_y**2)]
]) # process noise covariance matrix

R = np.array([
    [0, 0],
    [0, 1]
]) # measurement noise covariance matrix

X = np.array([
    [0],
    [0],
    [0],
    [0]
]) # initial state vector (we assumed starting position of 0 with 0 initial velocity)

kf = KalmanFilter(A, P, H, Q, R, X) # create a Kalman Filter initialized with our configurations
# ---


# -----
#   Run the simulation and report the results:
predicted_posisitons = []
updated_posisitons = []
for t, Y in enumerate(measurements):
    initial_X = kf.predict()
    updated_X = kf.update(Y)

    # logging
    print(f'time: {t}, real_position: {np.squeeze(locations[t], axis=1)}, noisy_position_measurement: {np.squeeze(measurements[t], axis=1)}, predicted_position: {np.squeeze(initial_X, axis=1)[:2]}, updated_position: {np.squeeze(updated_X, axis=1)[:2]}')
    predicted_posisitons.append(np.squeeze(initial_X, axis=1)[:2])
    updated_posisitons.append(np.squeeze(updated_X, axis=1)[:2])


plt.figure()
plt.title('Kalman Filter Sample Run')
plt.plot(locations[:,0], locations[:,1], label="real posisitons")
plt.plot(measurements[:,0], measurements[:,1], label=("noisy " if add_noise_flag else "") + "measurements")
# plt.plot(np.array(predicted_posisitons)[:,0], np.array(predicted_posisitons)[:,1], label="predicted positons")
plt.plot(np.array(updated_posisitons)[:,0], np.array(updated_posisitons)[:,1], label="updated positions")
plt.xlabel('Y Position')
plt.ylabel('X Position')
plt.legend()
plt.show()
# ---