from map import Map, Obstacle
from reference_path import ReferencePath
from simulator import Simulator
from mpc import MPC
from model import simple_bycicle_model
import do_mpc
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import time
import globals
sys.path.append('../../')


""" User settings: """
show_animation = True
store_results = False

# Load map file
map = Map(file_path='maps/sim_map.png', origin=[-1, -2],
          resolution=0.005)

# Specify waypoints
wp_x = [-0.75, -0.25, -0.25, 0.25, 0.25, 1.25, 1.25, 0.75, 0.75, 1.25,
        1.25, -0.75, -0.75, -0.25]
wp_y = [-1.5, -1.5, -0.5, -0.5, -1.5, -1.5, -1, -1, -0.5, -0.5, 0, 0,
        -1.5, -1.5]

# Specify path resolution
path_resolution = 0.05  # m / wp

# Create smoothed reference path
reference_path = ReferencePath(map, wp_x, wp_y, path_resolution,
                               smoothing_distance=5, max_width=0.23,
                               circular=True)

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
    map.add_obstacles([obs1, obs2, obs3, obs4, obs5, obs6, obs7,
                       obs8, obs9])

"""
Get configured do-mpc modules:
"""
vehicle = simple_bycicle_model(
    length=0.12, width=0.06, Ts=0.05, reference_path=reference_path)
model = vehicle.model

controller = MPC(vehicle)
mpc = controller.mpc

simulator = Simulator(vehicle).simulator

# Compute speed profile
ay_max = 4.0  # m/s^2
a_min = -0.1  # m/s^2
a_max = 0.5  # m/s^2
SpeedProfileConstraints = {'a_min': a_min, 'a_max': a_max,
                           'v_min': 0.0, 'v_max': 1.0, 'ay_max': ay_max}
vehicle.reference_path.compute_speed_profile(SpeedProfileConstraints)


def update_new_bound(mpc, model, ay_max):
    # Compute dynamic constraints on e_y
    ub, lb, _ = vehicle.reference_path.update_path_constraints(
        vehicle.wp_id + 1,
        globals.horizon,
        2 * vehicle.safety_margin,
        vehicle.safety_margin,
    )

    upper_e_y_1 = np.mean(ub)
    lower_e_y_1 = np.mean(lb)
    upper_e_y_2 = ub[-1]
    lower_e_y_2 = lb[-1]

    # Get curvature predictions from previous control signals
    kappa_pred = np.tan(np.array(mpc.data['_u', 'delta'][0])) / vehicle.length

    # Constrain maximum speed based on predicted car curvature
    upper_vel = np.sqrt(ay_max / (np.abs(kappa_pred) + 1e-12))

    # reset the boundaries
    controller.constraints_setup(
        vel_bound=[0.0, upper_vel], e_y_bound=[lower_e_y_1, upper_e_y_1, lower_e_y_2, upper_e_y_2], reset=True
    )


"""
Set initial state
"""

x0 = np.array([vehicle.reference_path.waypoints[0].x, vehicle.reference_path.waypoints[0].y,
               vehicle.reference_path.waypoints[0].psi, 0.3, 0])
mpc.x0 = x0
simulator.x0 = x0

# Use initial state to set the initial guess.
mpc.set_initial_guess()

"""
Setup graphic:
"""

fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data)
plt.ion()

"""
Run MPC main loop:
"""

k = 0
while globals.s < reference_path.length:
    vehicle.get_current_waypoint()
    print("======= wp_id ======== ", vehicle.wp_id)
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)
    controller.distance_update(x0)

    if show_animation:

        # Plot path and drivable area
        reference_path.show(wp=vehicle.current_waypoint)
        vehicle.show(x0)
        plt.axis('off')
        plt.pause(0.001)
        plt.show()

        # graphics.plot_results(t_ind=k)
        # graphics.plot_predictions(t_ind=k)
        # graphics.reset_axes()
        # plt.show()
        # plt.pause(0.01)

    k += 1

input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'lateral control')
