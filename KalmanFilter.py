import numpy as np

class KalmanFilter():
    def __init__(self, A, P, H, Q, R, X):
        """
        Kalman Filter implementation with no control inputs (B, u).

        Inputs:
            A (np.array) = state transition matrix of the process from state at time k to the state at time k+1 (sometimes denoted as Phi)
            P (np.array) = error covariance matrix
            H (np.array) = (measurement matrix) noiseless connection btw the state vector and the measurement vector
            Q (np.array) = process noise covariance matrix
            R (np.array) = measurement noise covariance matrix
            X (np.array) = initial state vector

        """
        self.A = A # state transition matrix of the process from state at time k to the state at time k+1 (sometimes denoted as Phi)
        self.P = P # error covariance matrix
        self.H = H # (measurement matrix) noiseless connection btw the state vector and the measurement vector
        self.Q = Q # process noise covariance matrix
        self.R = R # measurement noise covariance matrix

        self.X = X # state vector
    
    def predict(self):
        """
        Performs the predict step in the Kalman filter which simply propagates the previous state 
        estimate forward using the system's state transition model, while also updating the 
        uncertainty based on the process noise covariance.

        Returns:
            self.X: state vector estimate obtained by the prediction step.
        """
        # State Projection:
        # X_t = A X_{t-1}
        self.X = np.matmul(self.A, self.X)

        # Covariance Maatrix Projection
        # P_t = A P_{k-1} A^T + Q
        self.P = np.matmul(self.A, np.matmul(self.P, np.transpose(self.A))) + self.Q
        
        return self.X

    def update(self, Y):
        """
        The update step in the Kalman filter corrects the predicted state estimate by 
        incorporating new measurements, adjusting the estimate and its uncertainty based on 
        the measurement residual and the Kalman gain (K).

        Inputs:
            Y (np.array): measurement vector
        Returns:
            self.X (np.array): updated state vector estimate obtained after the update step.
        """
        # Calculate optimal Kalman Gain (K)
        # K_t = P_t H^T (H P H^T + R)^-1
        K = np.matmul(self.P, np.matmul(np.transpose(self.H), np.linalg.inv((np.matmul(self.H, np.matmul(self.P, np.transpose(self.H))) + self.R))))

        # Update estimate (X)
        # X_t = X_t + K_t (Y_t - H X_t)
        self.X = self.X + np.matmul(K, (Y - np.matmul(self.H, self.X)))

        # Update Error Covariance Matrix (P)
        # P_t = (I - K_t H) P_t
        self.P = np.matmul((np.identity(K.shape[0]) - np.matmul(K, self.H)), self.P)

        return self.X