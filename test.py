from KalmanFilter import KalmanFilter
import numpy as np
import unittest
from filterpy.kalman import KalmanFilter as FilterPyKalmanFilter

def get_2D_KalmanFilter():
    """ Helper method that returns a Kalman Filter initialized for 2D object tracking with state vector= [pos_x, pos_y, vel_x, vel_y], 
    and measurements= [pos_x, pos_y], with no control inputs (B, u)"""
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

    return kf



class TestKalmanFilter_2D_NoControlInput(unittest.TestCase):

    def setUp(self):
        self.kf = get_2D_KalmanFilter()
    
    def test_initialization(self):
        """ Test that the Kalman filter initializes with correct state and covariance """
        self.assertTrue(np.array_equal(self.kf.X, np.zeros((4,1))))
        self.assertTrue(np.array_equal(self.kf.P, np.eye(4) * 1000))
    
    def test_predict(self):
        """ Test the prediction step of the Kalman filter """
        initial_X = self.kf.X.copy()
        self.kf.predict()

        # The predicted state should evolve based on the system dynamics
        if np.sum(np.abs(self.kf.X[2:])) == 0: # if all velocities in the state vector are 0 then it should not move upon predicting
            # For simplicity, we expect the position to stay at [0,0] and the velocity at [0,0]
            # Assuming no control input and constant motion model.
            self.assertTrue(np.all(self.kf.X == initial_X))  # State should still be [0, 0, 0, 0] after the predict step with no motion
        else: # iff there is a non-zero velocity  in the state vector then the state vector should change after the prediction step
            self.assertFalse(np.array_equal(self.kf.X, initial_X))  # The state should have changed
    
    def test_covariance_update(self):
        """ Test the covariance update after a predict-update cycle """
        initial_P = self.kf.P.copy()

        # Perform one predict and update cycle
        self.kf.predict()
        measurement = np.array([[1], [1]])  # Example measurement
        self.kf.update(measurement)

        # The covariance matrix should decrease in uncertainty after an update
        self.assertTrue(np.all(self.kf.P.diagonal() < initial_P.diagonal()))  # Covariance should be smaller after update
    
    def test_multiple_updates(self):
        """ Test that the Kalman filter works correctly over multiple prediction and update cycles """
        for _ in range(10):
            self.kf.predict()
            measurement = np.array([1, 1])  # Example measurement
            self.kf.update(measurement)

        # After multiple updates, the state should have converged closer to the measurement [1,1]
        self.assertTrue(np.all(np.abs(self.kf.X[:2] - np.array([1, 1])) < 0.1))  # Position should be very close to [1, 1]

    def test_update(self):
        """ Test the update step of the Kalman filter with a dummy measurement """
        measurement = np.array([[1], [1]])  # Example measurement
        initial_X = self.kf.X.copy()

        # Update the Kalman filter with a measurement
        self.kf.update(measurement)

        # After update, the state should move closer to the measurement (since it's initially at [0,0])
        self.assertFalse(np.array_equal(self.kf.X, initial_X))  # The state should have been updated

        # Check that the updated state is closer to the measurement
        self.assertTrue(np.all((np.abs(self.kf.X[:2]) - measurement) < np.abs(initial_X[:2] - measurement)))  # Position should be closer to [1, 1]


class TestKalmanFilterByComparison(unittest.TestCase):
    """
    Compare the correctness of our Kalman Filter implementation by comparing its results against an already available
    Kalman Filter implementation from the FilterPY library.
    """

    def setUp(self):
        # Initial parameters for both filters
        initial_X = np.array([0, 0, 0, 0])  # Initial state (pos_x, pos_y, vel_x, vel_y)
        P = np.eye(4) * 1000  # High initial uncertainty
        Q = np.array([[1, 0, 0, 0],  # Process noise (Q)
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        R = np.array([[10, 0],  # Measurement noise (R)
                      [0, 10]])
        H = np.array([[1, 0, 0, 0],  # Measurement matrix (H)
                      [0, 1, 0, 0]])

        # Initialize FilterPy's Kalman Filter (will be used as the ground truth to evaluate the correctness of our implementation)
        self.kf_filterpy = FilterPyKalmanFilter(dim_x=4, dim_z=2)  # filterpy's KalmanFilter takes these args
        self.kf_filterpy.x = initial_X
        self.kf_filterpy.P = P
        self.kf_filterpy.Q = Q
        self.kf_filterpy.R = R
        self.kf_filterpy.H = H

        # Initialize our custom Kalman Filter
        self.kf_custom = KalmanFilter(self.kf_filterpy.F, P, H, Q, R, np.expand_dims(initial_X, axis=1)) # Note that filterpy's F corresponds to our kalman filter impelementation's A

    def test_single_update_step_comparison(self):
        """ Test both Kalman filters over a single prediction-update cycle by comparing state estimates and covariance """
        # Dummy measurement (position at [1, 1])
        measurement = np.array([1, 1])

        # Run one prediction and update cycle on both filters
        self.kf_custom.predict()
        self.kf_custom.update(np.expand_dims(measurement, axis=1))

        self.kf_filterpy.predict()
        self.kf_filterpy.update(measurement)

        # Compare state estimates (positions and velocities) using np.allclose for tolerance
        self.assertTrue(np.allclose(np.squeeze(self.kf_custom.X, axis=1), self.kf_filterpy.x, atol=1e-4),
                        "State estimates are not close enough")

        # Compare covariance matrices, check if they're close enough
        self.assertTrue(np.allclose(self.kf_custom.P, self.kf_filterpy.P, atol=1e-4),
                        "Covariance matrices are not close enough")
        
    def test_multiple_update_steps_comparison(self):
        """ Test both Kalman filters over multiple prediction-update cycles with different measurements by comparing state estimates and covariance """
        measurements = [np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4])]

        for measurement in measurements:
            # Run prediction and update for custom filter
            self.kf_custom.predict()
            self.kf_custom.update(np.expand_dims(measurement, axis=1))

            # Run prediction and update for filterpy filter
            self.kf_filterpy.predict()
            self.kf_filterpy.update(measurement)

            # Compare state estimates after each cycle
            self.assertTrue(np.allclose(np.squeeze(self.kf_custom.X, axis=1), self.kf_filterpy.x, atol=1e-4),
                            f"State estimates mismatch at measurement {measurement}")

            # Compare covariance matrices after each cycle
            self.assertTrue(np.allclose(self.kf_custom.P, self.kf_filterpy.P, atol=1e-4),
                            f"Covariance matrices mismatch at measurement {measurement}")
    
    def test_state_update(self):
        """ Test that both filters update their state correctly after the prediction-update cycle """
        initial_state_custom = self.kf_custom.X.copy()
        initial_state_filterpy = self.kf_filterpy.x.copy()

        # Update step with a measurement (position at [5, 5])
        measurement = np.array([5, 5])

        # Run prediction and update for both filters
        self.kf_custom.predict()
        self.kf_custom.update(np.expand_dims(measurement, axis=1))

        self.kf_filterpy.predict()
        self.kf_filterpy.update(measurement)

        # Ensure that the state after update has changed (not identical to initial state)
        self.assertFalse(np.array_equal(self.kf_custom.X, initial_state_custom),
                         "Custom Kalman filter state did not update correctly")
        self.assertFalse(np.array_equal(self.kf_filterpy.x, initial_state_filterpy),
                         "FilterPy Kalman filter state did not update correctly")

        # Check if both filters have similar state after the update
        self.assertTrue(np.allclose(np.squeeze(self.kf_custom.X, axis=1), self.kf_filterpy.x, atol=1e-4),
                        "State estimates mismatch after update")
