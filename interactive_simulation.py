import pygame
from KalmanFilter import KalmanFilter
import numpy as np


def get_KalmanFilter2D(fps):
    """
    Initialize a Kalman Filter for 2D (object) tracking with no control inputs, 
    internal state representation of [pos_x, pos_y, velocity_x, veloctiy_y], and measurements of [pos_x, pos_y].
    """

    dt = 1/fps # duration of discrete time steps (1/fps of the game)
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


if __name__ == '__main__':
    fps = 60 # frames per second of the event loop

    # Initialize pygame
    pygame.init()

    # Set the background color
    bg_color = (255, 255, 255)  # White background
    line_color_cursor = (0, 0, 0)  # Black line for the track
    line_color_kf = (255, 0, 0)  # Black line for the track

    # Set up the display
    width, height = 600, 400
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('2D Kalman Filter Mouse Tracking')
    font = pygame.font.Font('freesansbold.ttf', 15)
    text_mouse_legend = font.render('Mouse Position', True, line_color_cursor, bg_color)
    text_kf_legend = font.render('KF Prediction', True, line_color_kf, bg_color)
    # text_mouse_legend = font.render('KF Prediction', True, line_color_kf, bg_color)
    text_mouse_legend_rect = text_mouse_legend.get_rect()
    text_kf_legend_rect = text_mouse_legend.get_rect()
    text_mouse_legend_rect.center = (80, 10)
    text_kf_legend_rect.center = (80, 30)


    # List to store the path of the mouse and the KF predictions
    mouse_path = []
    kf_path = []
    kf = None

    # Main game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get the current mouse position
        mouse_x, mouse_y = pygame.mouse.get_pos()

        if kf == None: # initialize kalman filter for cursor tracking
            kf = get_KalmanFilter2D(fps)
            kf.X = np.array([
                [mouse_x],
                [mouse_y],
                [0],
                [0]
            ])

        kf.predict()
        kf_X = kf.update(np.array([
                [mouse_x],
                [mouse_y]
            ]))

        # Add the current mouse position to the path list
        mouse_path.append((mouse_x, mouse_y))

        # Add the kalman filter position prediction to the path list
        kf_path.append(np.squeeze(kf_X[:2], axis=1).tolist())

        # Clear the screen and fill with background color
        screen.fill(bg_color)
        screen.blit(text_mouse_legend, text_mouse_legend_rect)
        screen.blit(text_kf_legend, text_kf_legend_rect)

        # Draw the mouse path (lines between consecutive mouse positions)
        if len(mouse_path) > 1:
            pygame.draw.lines(screen, line_color_cursor, False, mouse_path, 10)
        if len(kf_path) > 1:
            pygame.draw.lines(screen, line_color_kf, False, kf_path, 10)

        # Update the display
        pygame.display.flip()

        # Control the frame rate
        pygame.time.Clock().tick(fps)

    # Quit pygame when the loop ends
    pygame.quit()
