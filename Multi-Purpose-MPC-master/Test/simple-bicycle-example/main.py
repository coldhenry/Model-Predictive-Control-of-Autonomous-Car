from reference_path import ReferencePath
from map import Map, Obstacle
from model import simple_bycicle_model
from simulator import Simulator
from MPC import MPC
import pdb
import sys
import time
import do_mpc
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import globals

sys.path.append('../../')


''' User settings: '''
show_animation = True
store_results = False

# Load map file
map = Map(file_path='maps/sim_map.png', origin=[-1, -2], resolution=0.005)

# Specify waypoints
wp_x = [-0.75, -0.25, -0.25, 0.25, 0.25, 1.25, 1.25, 0.75, 0.75, 1.25, 1.25, -0.75, -0.75, -0.25]
wp_y = [-1.5, -1.5, -0.5, -0.5, -1.5, -1.5, -1, -1, -0.5, -0.5, 0, 0, -1.5, -1.5]

# Specify path resolution
path_resolution = 0.05  # m / wp

# Create smoothed reference path
reference_path = ReferencePath(
    map,
    wp_x,
    wp_y,
    path_resolution,
    smoothing_distance=5,
    max_width=0.23,
    circular=True,
)

# Add obstacles
use_obstacles = False
if use_obstacles:
    obs1 = Obstacle(cx=0.0, cy=0.0, radius=0.05)
    obs2 = Obstacle(cx=-0.8, cy=-0.5, radius=0.08)
    obs3 = Obstacle(cx=-0.7, cy=-1.5, radius=0.05)
    obs4 = Obstacle(cx=-0.3, cy=-1.0, radius=0.08)
    obs5 = Obstacle(cx=0.27, cy=-1.0, radius=0.05)
    obs6 = Obstacle(cx=0.78, cy=-1.47, radius=0.05)
    obs7 = Obstacle(cx=0.73, cy=-0.9, radius=0.07)
    obs8 = Obstacle(cx=1.2, cy=0.0, radius=0.08)
    obs9 = Obstacle(cx=0.67, cy=-0.05, radius=0.06)
    map.add_obstacles([obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8, obs9])


'''
Get configured do-mpc modules:
'''
# model setup
vehicle = simple_bycicle_model(
    length=0.12, width=0.06, reference_path=reference_path, Ts=0.05
)
vehicle.model_setup()
model = vehicle.model

controller = MPC(vehicle)
mpc = controller.mpc

sim_instance = Simulator(vehicle)
simulator = sim_instance.simulator

# Compute speed profile
ay_max = 4.0  # m/s^2
a_min = -1  # m/s^2
a_max = 1  # m/s^2
SpeedProfileConstraints = {
    'a_min': a_min,
    'a_max': a_max,
    'v_min': 0.0,
    'v_max': 1.0,
    'ay_max': ay_max,
}
vehicle.reference_path.compute_speed_profile(SpeedProfileConstraints)


def update_new_bound(mpc, model, ay_max):
    # Compute dynamic constraints on e_y
    ub, lb, _ = vehicle.reference_path.update_path_constraints(
        vehicle.wp_id + 1,
        globals.horizon,
        2 * vehicle.safety_margin,
        vehicle.safety_margin,
    )

    upper_e_y = np.mean(ub)
    lower_e_y = np.mean(lb)

    # Get curvature predictions from previous control signals
    kappa_pred = np.tan(np.array(mpc.data['_u', 'delta'][0])) / vehicle.length

    # Constrain maximum speed based on predicted car curvature
    upper_vel = np.sqrt(ay_max / (np.abs(kappa_pred) + 1e-12))

    # reset the boundaries
    controller.constraints_setup(
        vel_bound=[0.0, upper_vel], e_y_bound=[lower_e_y, upper_e_y], reset=True
    )


'''
Set initial state
'''
x0 = np.array([vehicle.reference_path.waypoints[0].x, vehicle.reference_path.waypoints[0].y, 0, 0])
mpc.x0 = x0
simulator.x0 = x0

# Use initial state to set the initial guess.
mpc.set_initial_guess()


'''
Run MPC main loop:
'''

##############
# Simulation #
##############

# Set simulation time to zero
t = 0.0

# Logging containers
x_log = [x0[0]]
y_log = [x0[0]]
v_log = [0.0]

# Until arrival at end of path
# vehicle.s < reference_path.length
while 1:
    # Get control signals
    u = controller.get_control(x0)

    # DEBUG
    # print("=========================")
    # print("delta: ", u[1])
    # print("psi: ", x0[2])
    # print("psi_ref: ", vehicle.reference_path.waypoints[vehicle.wp_id].psi)
    # print("x: ", x0[0])
    # print("x_ref: ", vehicle.reference_path.waypoints[vehicle.wp_id].x)
    # print("=========================")

    # Simulate car
    x0 = simulator.make_step(u)
    controller.distance_update(x0)

    # Log car state
    x_log.append(x0[0])
    y_log.append(x0[1])
    v_log.append(x0[3])

    # Increment simulation time
    t += vehicle.Ts

    # Plot path and drivable area
    reference_path.show(id=vehicle.wp_id)

    # Plot car
    sim_instance.show(x0)

    # Plot MPC prediction
    controller.show_prediction()

    # update boundary for the next iteration
    update_new_bound(mpc, model, ay_max)

    # Set figure title
    # plt.title('MPC Simulation: acc(t): {:.2f}, delta(t): {:.2f}, Duration: '
    #           '{:.2f} s'.format(u[0], u[1], t))
    plt.axis('off')
    plt.pause(0.001)

input('Press any key to exit.')
