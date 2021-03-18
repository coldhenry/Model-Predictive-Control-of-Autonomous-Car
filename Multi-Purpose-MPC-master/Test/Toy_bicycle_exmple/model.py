import do_mpc
import numpy as np
from casadi import *
from casadi.tools import *
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
import math
import pdb
import sys
import globals
sys.path.append("../../")

# Colors
PREDICTION = '#BA4A00'
CAR = '#F1C40F'
CAR_OUTLINE = '#B7950B'


class simple_bycicle_model:
    def __init__(self, length, width, Ts, reference_path):

        # car paramters
        self.length = length
        self.width = width

        # model
        self.Ts = Ts
        self.model = None

        # waypoint
        self.reference_path = reference_path
        self.wp_id = 0
        self.current_waypoint = self.reference_path.waypoints[self.wp_id]

        self.model_setup()

    def model_setup(self):

        model_type = 'continuous'  # either 'discrete' or 'continuous'
        self.model = do_mpc.model.Model(model_type)

        # States struct (optimization variables):
        pos_x = self.model.set_variable(var_type='_x', var_name='pos_x')
        pos_y = self.model.set_variable(var_type='_x', var_name='pos_y')
        psi = self.model.set_variable(var_type='_x', var_name='psi')
        vel = self.model.set_variable(var_type='_x', var_name='vel')
        e_y = self.model.set_variable(var_type='_x', var_name='e_y')

        # Input struct (optimization variables):
        acc = self.model.set_variable(var_type='_u', var_name='acc')
        delta = self.model.set_variable(var_type='_u', var_name='delta')

        # setpoint
        ref_x = self.model.set_variable(var_type='_tvp', var_name='ref_x')
        ref_y = self.model.set_variable(var_type='_tvp', var_name='ref_y')
        ref_psi = self.model.set_variable(var_type='_tvp', var_name='ref_psi')

        # tracking errors (optimization variables):
        # difference of two psi angle needs to consider the edge case around -2pi and 2p
        e_psi_current = (fmod(psi - ref_psi + np.pi, 2 * np.pi) - np.pi)
        e_psi_next = (fmod(psi - ref_psi + np.pi, 2 * np.pi) - np.pi) + \
            vel * delta / self.length * self.Ts

        self.model.set_expression('e_psi_next', e_psi_next)
        self.model.set_expression('e_psi_current', e_psi_current)

        self.model.set_rhs('pos_x', vel * cos(psi))
        self.model.set_rhs('pos_y', vel * sin(psi))
        self.model.set_rhs('psi', vel / self.length * (delta))
        self.model.set_rhs('vel', acc)
        self.model.set_rhs('e_y', vel * sin(e_psi_current))

        self.model.setup()

    def _compute_safety_margin(self):
        """
        Compute safety margin for car if modeled by its center of gravity.
        """
        # Model ellipsoid around the car
        safety_margin = self.width / np.sqrt(2)

        return safety_margin

    def get_current_waypoint(self):
        """
        Get closest waypoint on reference path based on car's current location.
        """

        # Compute cumulative path length
        length_cum = np.cumsum(self.reference_path.segment_lengths)
        # Get first index with distance larger than distance traveled by car
        # so far

        greater_than_threshold = length_cum > globals.s
        next_wp_id = greater_than_threshold.searchsorted(True)
        # Get previous index
        prev_wp_id = next_wp_id - 1

        # Get distance traveled for both enclosing waypoints
        s_next = length_cum[next_wp_id]
        s_prev = length_cum[prev_wp_id]

        if np.abs(globals.s - s_next) < np.abs(globals.s - s_prev):
            self.wp_id = next_wp_id
            self.current_waypoint = self.reference_path.waypoints[next_wp_id]
        else:
            self.wp_id = prev_wp_id
            self.current_waypoint = self.reference_path.waypoints[prev_wp_id]

    def show(self, states):
        '''
        Display car on current axis.
        '''
        x, y, psi = states[0], states[1], states[2]

        # Get car's center of gravity
        cog = (x, y)
        # Get current angle with respect to x-axis
        yaw = np.rad2deg(psi)
        # Draw rectangle
        car = plt_patches.Rectangle(
            cog,
            width=self.length,
            height=self.width,
            angle=yaw,
            facecolor=CAR,
            edgecolor=CAR_OUTLINE,
            zorder=20,
        )

        # Shift center rectangle to match center of the car
        car.set_x(
            car.get_x()
            - (
                self.length / 2 * np.cos(psi)
                - self.width / 2 * np.sin(psi)
            )
        )
        car.set_y(
            car.get_y()
            - (
                self.width / 2 * np.cos(psi)
                + self.length / 2 * np.sin(psi)
            )
        )

        # Add rectangle to current axis
        ax = plt.gca()
        ax.add_patch(car)
