import do_mpc
import numpy as np
from casadi import *
from casadi.tools import *
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
import math
import pdb
import sys

from map import Map, Obstacle
from reference_path import ReferencePath

sys.path.append("../../")

# Colors
CAR = '#F1C40F'
CAR_OUTLINE = '#B7950B'


class simple_bycicle_model:
    def __init__(self, reference_path, length, width, Ts):

        # car paramters
        self.length = length
        self.width = width

        # reference and safety
        self.safety_margin = self._compute_safety_margin()
        self.reference_path = reference_path

        # waypoint
        self.wp_id = 0
        self.current_waypoint = self.reference_path.waypoints[self.wp_id]

        # distance traveled
        self.s = 0.0

        # model
        self.Ts = Ts
        self.model = None

    def model_setup(self):

        model_type = 'discrete'  # either 'discrete' or 'continuous'
        self.model = do_mpc.model.Model(model_type)

        # States struct (optimization variables):
        pos_x = self.model.set_variable(var_type='_x', var_name='pos_x')
        pos_y = self.model.set_variable(var_type='_x', var_name='pos_y')
        psi = self.model.set_variable(var_type='_x', var_name='psi')
        vel = self.model.set_variable(var_type='_x', var_name='vel')
        e_y = self.model.set_variable(var_type='_x', var_name='e_y')
        e_psi = self.model.set_variable(var_type='_x', var_name='e_psi')

        # Input struct (optimization variables):
        acc = self.model.set_variable(var_type='_u', var_name='acc')
        delta = self.model.set_variable(var_type='_u', var_name='delta')

        # reference data
        # using time-varing parameters data type
        x_ref = self.model.set_variable(var_type='_tvp', var_name='x_ref')
        y_ref = self.model.set_variable(var_type='_tvp', var_name='y_ref')
        psi_ref = self.model.set_variable(var_type='_tvp', var_name='psi_ref')
        vel_ref = self.model.set_variable(var_type='_tvp', var_name='vel_ref')

        self.model.set_rhs('pos_x', vel * cos(psi))
        self.model.set_rhs('pos_y', vel * sin(psi))
        self.model.set_rhs('psi', vel / self.length * tan(delta))
        self.model.set_rhs('vel', acc)
        self.model.set_rhs('e_y', vel * sin(e_y))
        self.model.set_rhs('e_psi', vel / self.length * tan(delta))

        self.model.setup()
        return self.model

    def drive(self):
        """
        Drive.
        :param u: input vector containing [v, delta]
        """

        # Compute velocity along path
        # TODO: need to confirm the equation
        s_dot = self.model.x['vel'] * np.cos(self.model.x['e_psi'])

        # Update distance travelled along reference path
        self.s += s_dot * self.Ts

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
        greater_than_threshold = length_cum > self.s
        next_wp_id = greater_than_threshold.searchsorted(True)
        # Get previous index
        prev_wp_id = next_wp_id - 1

        # Get distance traveled for both enclosing waypoints
        s_next = length_cum[next_wp_id]
        s_prev = length_cum[prev_wp_id]

        if np.abs(self.s - s_next) < np.abs(self.s - s_prev):
            self.wp_id = next_wp_id
            self.current_waypoint = self.reference_path.waypoints[next_wp_id]
        else:
            self.wp_id = prev_wp_id
            self.current_waypoint = self.reference_path.waypoints[prev_wp_id]

    def show(self):
        """
        Display car on current axis.
        """

        # Get car's center of gravity
        cog = (self.model.x['pos_x'], self.model.x['pos_y'])
        # Get current angle with respect to x-axis
        yaw = np.rad2deg(self.model.x['psi'])
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
                self.length / 2 * np.cos(self.model.x['psi'])
                - self.width / 2 * np.sin(self.model.x['psi'])
            )
        )
        car.set_y(
            car.get_y()
            - (
                self.width / 2 * np.cos(self.model.x['psi'])
                + self.length / 2 * np.sin(self.model.x['psi'])
            )
        )

        # Add rectangle to current axis
        ax = plt.gca()
        ax.add_patch(car)
