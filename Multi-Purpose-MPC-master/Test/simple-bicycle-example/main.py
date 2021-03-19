import pdb
import sys
import time
import do_mpc
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *

from reference_path import ReferencePath
from map import Map, Obstacle
from model import simple_bycicle_model
from simulator import Simulator
from MPC import MPC

import globals

sys.path.append('../../')

print("hi")


def environment_setup(map_file, use_obstacles=True):

    # Load map file
    map = Map(file_path=map_file, origin=[-1, -2], resolution=0.005)

    # Specify waypoints
    wp_x = [-0.75, -0.25, -0.25, 0.25, 0.25, 1.25, 1.25,
            0.75, 0.75, 1.25, 1.25, -0.75, -0.75, -0.25]
    wp_y = [-1.5, -1.5, -0.5, -0.5, -1.5, -
            1.5, -1, -1, -0.5, -0.5, 0, 0, -1.5, -1.5]

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
    if use_obstacles:
        obs1 = Obstacle(cx=0.0, cy=0.05, radius=0.05)
        obs2 = Obstacle(cx=-0.85, cy=-0.5, radius=0.08)
        obs3 = Obstacle(cx=-0.75, cy=-1.5, radius=0.05)
        obs4 = Obstacle(cx=-0.35, cy=-1.0, radius=0.08)
        obs5 = Obstacle(cx=0.35, cy=-1.0, radius=0.05)
        obs6 = Obstacle(cx=0.78, cy=-1.47, radius=0.05)
        obs7 = Obstacle(cx=0.73, cy=-0.9, radius=0.07)
        obs8 = Obstacle(cx=1.2, cy=0.0, radius=0.08)
        obs9 = Obstacle(cx=0.67, cy=-0.05, radius=0.06)
        map.add_obstacles(
            [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8, obs9])

    return reference_path


def MPC_Problem_setup(reference_path, ay_max=4.0, a_min=-1, a_max=1):
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
    SpeedProfileConstraints = {
        'a_min': a_min,
        'a_max': a_max,
        'v_min': 0.0,
        'v_max': 1.0,
        'ay_max': ay_max,
    }
    vehicle.reference_path.compute_speed_profile(SpeedProfileConstraints)

    return vehicle, model, controller, mpc, sim_instance, simulator


if __name__ == '__main__':

    ''' User settings: '''
    show_animation = True
    store_results = False

    map_file = 'maps/sim_map.png'

    # set up the map/ assign the waypoints/ add obstacles
    reference_path = environment_setup(map_file, use_obstacles=True)

    # initiate the class of mpc, simulator
    vehicle, model, controller, mpc, sim_instance, simulator = MPC_Problem_setup(
        reference_path)

    '''
    Set initial state
    '''
    x0 = np.array([vehicle.reference_path.waypoints[0].x, vehicle.reference_path.waypoints[0].y,
                   vehicle.reference_path.waypoints[0].psi, 0.3, 0])
    mpc.x0 = x0
    simulator.x0 = x0

    # Use initial state to set the initial guess.
    mpc.set_initial_guess()

    '''
    Run MPC main loop:
    '''
    # Until arrival at end of path
    while globals.s < reference_path.length:
        # Get control signals
        u = controller.get_control(x0)

        # Simulate car
        x0 = simulator.make_step(u)
        controller.distance_update(x0)

        # Plot path and drivable area/ plot car
        reference_path.show(id=vehicle.wp_id)
        sim_instance.show(x0)

        # update boundary for the next iteration
        controller.constraints_setup()

        plt.axis('off')
        plt.pause(0.001)

    input('Press any key to exit.')
