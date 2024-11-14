# Kalman Filter derivation and implementation

This repository contains detailed hand derivations of the Kalman Filter (KF) equations. Implements a basic Kalman Filter in python, and using that implementation runs static and interactive tracking simulations. I hope this repository can be a good resource for others who want to follow the derivation of the Kalman Filter equations in full detail like I did when I first started to learn about this algorithm. The easy python simulations are meant to complement the hand derivations where the hand derivations cover the theoretical background, and the python simulations showcase how the derived equations can be applied to solve tracking problems.

## Contents:
* Step by step hand derivations of the Kalman Filter equations (see _Kalman Filter formula derivations.py_). These notes are in my own hand writing and not in something like Latex as I believed that this format allowed me to show connetions between different steps in a more visually accessable way. My derivations follow from the [Tutorial: The Kalman Filter](https://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf). One can start reading from the given resource and refer to my notes for a more detailed explanation of the steps in between.

* A very easy to follow implementation of the Kalman Filter (see _KalmanFilter.py_).

* a static simulation in python for 2D (object) pose tracking with no control inputs (see _static_simulation.py_).

        internal state representation of [pos_x, pos_y, velocity_x, veloctiy_y], and measurement readings of [pos_x, pos_y].
    ---
    __Running:__ <br>
    ```bash
    python static_simulation.py
    ```
    <br>    
    <img src="./assets/static simulation ss.png" width="60%" title="static simulation ss">
        

* An interactive simulation of 2D pose tracking where the Kalman Filter is used to estimate the pose of your cursor on the pygame window, and the estimated path is drawn on the screen along with the real path (see _interactive_simulation.py_).

        internal state representation of [pos_x, pos_y, velocity_x, veloctiy_y], and measurement readings of [pos_x, pos_y].
    ---
    __Running:__ <br>
    ```bash
    python interactive_simulation.py
    ```
    <br>    
    <img src="./assets/interactive simulation ss.png" width="60%" title="interactive simulation ss">

## What is a Kalman Filter ?
A __Kalman filter__ is an algorithm used to estimate the state of a dynamic system from noisy observations. It works by recursively combining predictions from a mathematical model (system dynamics) with new measurements, updating the estimate in a way that minimizes uncertainty. The filter assumes that both the system's process and measurement noise are Gaussian (normally distributed).

The Kalman filter operates in two main steps:
1. __Prediction__: The next state of the system is predicted based on the previous state and the system model.
2. __Correction (Update)__: The prediction is corrected using new measurements, adjusting the estimate to minimize the impact of measurement noise.

It’s widely used in areas like navigation, control systems, and robotics where sensor data is noisy and incomplete.

&emsp;__Required Math:__ <br>
* Probability
* Linear Algebra
* Matrix Calculus


## How to choose KF parameters? An example using a common application for 2D object tracking:
Kalman filter is commonly used for tracking detected objects in 2D images. For a single object tracking task, one can simply use Kalman Filter alone. But for the task of multi-object tracking, usually _Kalman Filter_ is used with another algorithm called the _Hungarian Algorithm_. This approach gives rise to a very common and simple multi-object tracking algorithm called the _SORT (Simple Online Realtime Tracking)_ algorithm.

Let's assume that we have an object detector which gives us the labels and the bounding boxes for the detected objects for a given frame, and we want to use the Kalman Filter to keep track of the position of the objects, what would be one way to choose the parameters of the Kalman Filter ?

We can choose the 2D state vector ($X_{t}$) so that it tracks the position in the x,y plane ($X$, $Y$), and its velocities in the x,y direction ($\dot{X}$, $\dot{Y}$). Assume that we get measurements ($Y_{t}$) consisting of the position in the x,y plane ($X$, $Y$). Writing in the matrix notation we have,

$$
X_{t} = \begin{bmatrix}X \\\ Y \\\ \dot{X} \\\ \dot{Y} \end{bmatrix} = A X_{t-1} + w
$$

$$
Y_{t} = \begin{bmatrix}X \\\ Y \end{bmatrix} = H X_{t} + v
$$

Now, recall the basic relation between position and velocity given by: <br>
new_position = old_position + velocity * $\Delta{t}$ +  $\frac{1}{2}$ *acceleration * $\Delta{t}^{2}$

new_velocity = old_velocity + acceleration * $\Delta{t}$

Assuming no acceleration, these equations reduce to: <br>

new_position = old_position_x + velocity * $\Delta{t}$

new_velocity = old_velocity

If we write our state ($X_{t}$) following this model we obtain:
```math
X_{t} = \begin{bmatrix}X \\\ Y \\\ \dot{X} \\\ \dot{Y} \end{bmatrix} = \begin{bmatrix}X+\dot{X} \Delta{t} \\\ Y + \dot{Y} \Delta{t}\\\ \dot{X} \\\ \dot{Y} \end{bmatrix} = A X_{t-1} = \underbrace{\begin{bmatrix}1 & 0 & \Delta{t} & 0 \\\ 0 & 1 & 0 & \Delta{t} \\\ 0 & 0 & 1 & 0 \\\ 0 & 0 & 0 & 1 \end{bmatrix}}_{A} \begin{bmatrix}X \\\ Y \\\ \dot{X} \\\ \dot{Y} \end{bmatrix}
```
Hence we found _state transition matrix_ $A$.

Now write our 2D measurement vector ($Y_{t}$) following this model which yields:
```math
Y_{t} = \begin{bmatrix} X \\\ Y\end{bmatrix} = H X_{t} = \underbrace{\begin{bmatrix} 1 & 0 & 0 & 0 \\\ 0 & 1 & 0 & 0 \end{bmatrix}}_{H} \begin{bmatrix} X \\\ Y \\\ \dot{X} \\\ \dot{Y}\end{bmatrix}
```
Hence we found $H$.

To find the _process noise covariance matrix_ ($Q$) we write out what it is:

$$
Q = E[w_k {w_k}^T] = E[\begin{bmatrix} \sigma_x \\\ \sigma_y \\\ \sigma_{\dot{x}} \\\ \sigma{\dot{y}} \end{bmatrix} \begin{bmatrix} \sigma_x & \sigma_y & \sigma_{\dot{x}} & \sigma_{\dot{y}} \end{bmatrix}] = \begin{bmatrix} {\sigma_x}^2 & \sigma_{xy} & \sigma_{x\dot{x}} & \sigma_{x\dot{y}} \\\ \sigma_{xy} & {\sigma_y}^2 & \sigma_{y\dot{x}} & \sigma_{y\dot{y}} \\\ \sigma_{x\dot{x}} & \sigma_{y\dot{x}} & {\sigma_{\dot{x}}}^2 & \sigma_{\dot{x}\dot{y}} \\\ \sigma_{x\dot{y}} & \sigma_{y\dot{y}} & \sigma_{\dot{x}\dot{y}} & {\sigma_{\dot{y}}}^2 \end{bmatrix}
$$

where $w_k$ is the process noise vector, <br>
 ${\sigma_x}^2, {\sigma_y}^2$ are the variances of noise terms for the position in the x and y directions,  <br>
 ${\sigma_{\dot{x}}}^2, {\sigma_{\dot{y}}}^2$ are the variances of the noise terms for the velocity in the x and y directions, <br>
$\sigma_{xy}, \sigma_{yx}, \sigma_{\dot{x}y}, \sigma_{y\dot{x}}, \sigma_{\dot{y}x}, \sigma_{x\dot{y}}$ are the covariances between the pairs of corresponding noise terms.

Let $\ddot{\sigma_x}$ be the acceleration in the x direction. Since we did not model acceleration in our process, this randomness in acceleration contributes to the system noise in the system. <br>
As

```math
X_t = \underbrace{X_{t-1} + \dot{X_{t-1}} \Delta{t}}_{\text{we only modeled this}} + \underbrace{\frac{1}{2} \ddot{X_{t-1}} {\Delta{t}}^2}_{\text{This is noise now}}
```

then;

$$
\dot{x} = \ddot{\sigma_x} \Delta{t}
$$

$$
\sigma_x = \frac{1}{2} \ddot{\sigma_x} {\Delta_{t}}^2
$$

to simplify take $\ddot{\sigma_x} = \ddot{\sigma_y} = 1$ which yields:

$$
\dot{x} = \ddot{\sigma_x} \Delta{t} = \Delta{t}
$$

$$
\sigma_x = \frac{1}{2} \ddot{\sigma_x} {\Delta_{t}}^2 = \frac{1}{2} {\Delta{t}}^2
$$

then by plugging these expressions into the $Q$ formula we obtained earlier, and by using the simplifications we get:

```math
Q = E[w_k {w_k}^T] = E[\begin{bmatrix} \sigma_x \\\ \sigma_y \\\ \sigma_{\dot{x}} \\\ \sigma{\dot{y}} \end{bmatrix} \begin{bmatrix} \sigma_x & \sigma_y & \sigma_{\dot{x}} & \sigma_{\dot{y}} \end{bmatrix}] = \begin{bmatrix} {\sigma_x}^2 & \sigma_{xy} & \sigma_{x\dot{x}} & \sigma_{x\dot{y}} \\\ \sigma_{xy} & {\sigma_y}^2 & \sigma_{y\dot{x}} & \sigma_{y\dot{y}} \\\ \sigma_{x\dot{x}} & \sigma_{y\dot{x}} & {\sigma_{\dot{x}}}^2 & \sigma_{\dot{x}\dot{y}} \\\ \sigma_{x\dot{y}} & \sigma_{y\dot{y}} & \sigma_{\dot{x}\dot{y}} & {\sigma_{\dot{y}}}^2 \end{bmatrix} = \begin{bmatrix} \frac{{\Delta{t}}^2 }{4} {\ddot{\sigma_x}}^2 & 0 & \frac{{\Delta{t}}^3 }{2} {\ddot{\sigma_x}}^2 & 0 \\\ 0 & \frac{{\Delta{t}}^2 }{4} {\ddot{\sigma_y}}^2 & 0 & \frac{{\Delta{t}}^3 }{2} {\ddot{\sigma_y}}^2 \\\ \frac{{\Delta{t}}^3 }{2} {\ddot{\sigma_x}}^2 & 0 & {\Delta{t}}^2 {\ddot{\sigma_x}}^2 & 0 \\\ 0 & \frac{{\Delta{t}}^2 }{4} {\ddot{\sigma_y}}^2 & 0 & {\Delta{t}}^2 {\ddot{y}}^2 \end{bmatrix}
```

```math
= \begin{bmatrix} \frac{{\Delta{t}}^4}{4} & 0 & \frac{{\Delta{t}}^3}{2} & 0 \\\ 0 & \frac{{\Delta{t}}^4}{4} & 0 & \frac{{\Delta{t}}^3}{2} \\\ \frac{{\Delta{t}}^3}{2} & 0 & {\Delta{t}}^2 & 0 \\\ 0 & \frac{{\Delta{t}}^3}{2} & 0 & {\Delta{t}}^2 \end{bmatrix}
```

 Once we have the $\Delta{t}$, we just plug it into the matrix above to have our $Q$ matrix. For example, if we are performing object detection on a video stream with 60 FPS (Frames Per Second), we could choose $\Delta{t} = \frac{1}{\text{FPS}} = \frac{1}{60}$.

 To find the measurement noise covariance matrix ($R$), we write out what it is:

 $$
 R = E[v_k {v_k}^T] = E[\begin{bmatrix} \hat{\sigma_x} \\\ \hat{\sigma_y}\end{bmatrix} \begin{bmatrix} \hat{\sigma_x} & \hat{\sigma_y}\end{bmatrix}] = \begin{bmatrix} {\hat{\sigma_x}}^2 & \hat{\sigma_{xy}} \\\ \hat{\sigma_{yx}} & {\hat{\sigma_y}}^2 \end{bmatrix} = \underbrace{\begin{bmatrix} 1 & 0 \\\ 0 & 1 \end{bmatrix}}_{\text{off diagonal elements are zero because we assumed no correlation between the measurement noise in the x and y directions}}
 $$

where $v_k$ is the measurement noise vector, <br>
 $\hat{\sigma_x}^2, \hat{\sigma_y}^2$ are the variances of noise terms for the position measurements in the x and y directions,  <br>
$\hat{\sigma_{xy}}, \hat{\sigma_{yx}}$ are the covariances between the pairs of position measurement noise in the x direction and the position measurement noise in the y direction.
 

To set the initial value of the _state (error) covariance matrix_ ($P_0$), start out by writing the formula for $P$:

```math
P_t = E[\underbrace{(x_t - \hat{x_t})}_{e_t} \underbrace{(x_t - \hat{x_t})^T}_{{e_t}^T}] = E[\begin{bmatrix} e_1 \\\ e_2 \\\ ... \\\ e_n \end{bmatrix} \begin{bmatrix} e_1 & e_2 & ... & e_n \end{bmatrix}] = E[\begin{bmatrix} {e_1}^2 & ... & e_1 e_n \\\ ... & ... & ... \\\ e_n e_1 & ... & {e_n}^2 \end{bmatrix}] = \begin{bmatrix} {\text{var}(e_1}^2) & ... & \text{cov}(e_1 e_n) \\\ ... & ... & ... \\\ \text{cov}(e_n e_1) & ... & \text{var}(e_n) \end{bmatrix}
```

In our formulation of the system $X_t$ is a four dimensional vector (i.e. 4x1), which results in $e_t$ being a four dimensional vector as follows:

$$
e_t = \begin{bmatrix} e_1 \\\ e_2 \\\ e_3 \\\ e_4 \end{bmatrix}
$$

Then it results in 4x4 dimensional $P$ matrix which we can choose to initialize $P$ as:

$$
P_0 = \begin{bmatrix} 1000 & 0 & 0 & 0 \\\ 0 & 1000 & 0 & 0 \\\ 0 & 0 & 1000 & 0 \\\  0 & 0 & 0 & 1000 \end{bmatrix}
$$

The initial covariance matrix $P_0$ represents our initial uncertainty in the state estimate. In this case, we initialized it with high uncertainty in the state, assuming that we have little information about the object's initial position and velocity. Note that $P_0 = I \times 1000$ where $I$ is the identity matrix, and $1000$ is a large value reflecting the initial uncertainty. This ensures that the Kalman Filter starts with significant confidence in the measurements but adjusts over time as the filter runs. Note that the off-diagonal terms are all set to $0$, as we assume any two elements of the $e_t$ vector to be independent random variables which results in their covariance values to be $0$ as two independent random variables are guaranteed to have a covariance of $0$.

---