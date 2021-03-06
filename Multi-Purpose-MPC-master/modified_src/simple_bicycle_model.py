import numpy as np
from abc import abstractmethod
from abc import ABC
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
import math

##################################################################
#                                                                #
#   Based on Multi-Purpose MPC from Mats Steinweg, ZTH           #
#   Github: https://github.com/matssteinweg/Multi-Purpose-MPC    #
#                                                                #
##################################################################


# Colors
CAR = '#F1C40F'
CAR_OUTLINE = '#B7950B'

#########################
# Temporal State Vector #
#########################


class TemporalState:
    def __init__(self, x=0.0, y=0.0, psi=0.0):
        """
        Temporal State Vector containing car pose (x, y, psi)
        :param x: x position in global coordinate system | [m]
        :param y: y position in global coordinate system | [m]
        :param psi: yaw angle | [rad]
        """
        self.x = x
        self.y = y
        self.psi = psi

        self.members = ['x', 'y', 'psi']

    def __iadd__(self, other):
        """
        Overload Sum-Add operator.
        :param other: numpy array to be added to state vector
        """
        for state_id in range(len(self.members)):
            vars(self)[self.members[state_id]] += other[state_id]
        return self

    def __len__(self):
        return len(self.members)

    def __setitem__(self, key, value):
        vars(self)[self.members[key]] = value

    def __getitem__(self, item):
        if isinstance(item, int):
            members = [self.members[item]]
        else:
            members = self.members[item]
        return [vars(self)[key] for key in members]

    def list_states(self):
        """
        Return list of names of all states.
        """
        return self.members


###################################
# Simple Bicycle Model Base Class #
###################################


class BicycleModel(ABC):
    def __init__(self, reference_path, length, width, Ts):
        """
        Abstract Base Class for Bicycle Model.
        :param reference_path: reference path object to follow
        :param length: length of car in m
        :param width: width of car in m
        :param Ts: sampling time of model
        """

        # Precision
        self.eps = 1e-12

        # Car Parameters
        self.length = length
        self.width = width
        self.safety_margin = self._compute_safety_margin()

        # Reference Path
        self.reference_path = reference_path

        # Set initial distance traveled
        self.s = 0.0

        # Set sampling time
        self.Ts = Ts

        # Set initial waypoint ID
        self.wp_id = 0

        # Set initial waypoint
        self.current_waypoint = self.reference_path.waypoints[self.wp_id]

        # Declare temporal state variable | Initialization in sub-class
        self.temporal_state = None

    def drive(self, u):
        """
        Drive.
        :param u: input vector containing [v, delta]
        """

        # Get input signals
        v, delta = u

        # Compute temporal state derivatives
        x_dot = v * np.cos(self.temporal_state.psi)
        y_dot = v * np.sin(self.temporal_state.psi)
        psi_dot = v / self.length * np.tan(delta)
        temporal_derivatives = np.array([x_dot, y_dot, psi_dot])

        # Update spatial state (Forward Euler Approximation)
        self.temporal_state += temporal_derivatives * self.Ts

        # Compute velocity along path
        # TODO: need to confirm the equation
        s_dot = v * np.cos(delta)

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
        cog = (self.temporal_state.x, self.temporal_state.y)
        # Get current angle with respect to x-axis
        yaw = np.rad2deg(self.temporal_state.psi)
        # Draw rectangle
        car = plt_patches.Rectangle(cog, width=self.length, height=self.width,
                                    angle=yaw, facecolor=CAR,
                                    edgecolor=CAR_OUTLINE, zorder=20)

        # Shift center rectangle to match center of the car
        car.set_x(car.get_x() - (self.length / 2 *
                                 np.cos(self.temporal_state.psi) -
                                 self.width / 2 *
                                 np.sin(self.temporal_state.psi)))
        car.set_y(car.get_y() - (self.width / 2 *
                                 np.cos(self.temporal_state.psi) +
                                 self.length / 2 *
                                 np.sin(self.temporal_state.psi)))

        # Add rectangle to current axis
        ax = plt.gca()
        ax.add_patch(car)

    @abstractmethod
    def linearize(self, v_ref, psi_ref, delta_ref):
        pass


########################
# Simple Bicycle Model #
########################

class SimpleBicycleModel(BicycleModel):

    def __init__(self, reference_path, length, width, Ts):

        super(SimpleBicycleModel, self).__init__(
            reference_path, length=length, width=width, Ts=Ts)

        self.temporal_state = TemporalState()

        self.temporal_state.x = self.current_waypoint.x
        self.temporal_state.y = self.current_waypoint.y
        self.temporal_state.psi = self.current_waypoint.psi

        # Number of spatial state variables
        self.n_states = len(self.temporal_state)

    def linearize(self, v_ref, psi_ref, delta_ref):
        """
        Linearize the system equations around provided reference values.
        :param v_ref: velocity reference around which to linearize
        :param kappa_ref: kappa of waypoint around which to linearize
        :param delta_s: distance between current waypoint and next waypoint
         """

        ###################
        # System Matrices #
        ###################

        # Construct Jacobian Matrix
        # TODO
        a_1 = np.array([0, 0, -v_ref * np.sin(psi_ref)])
        a_2 = np.array([0, 0, v_ref * np.cos(psi_ref)])
        a_3 = np.array([0, 0, 0])

        b_1 = np.array([np.cos(psi_ref), 0])
        b_2 = np.array([np.sin(psi_ref), 0])
        b_3 = np.array([np.tan(delta_ref)/self.length, v_ref *
                        (np.tan(delta_ref)**2 + 1)/self.length])

        # TODO: duuno what this is
        # f = np.array([0.0, 0.0, 1 / v_ref * delta_s])

        A = np.stack((a_1, a_2, a_3), axis=0)
        B = np.stack((b_1, b_2, b_3), axis=0)

        return A, B
