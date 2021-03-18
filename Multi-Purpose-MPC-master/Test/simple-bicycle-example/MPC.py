import do_mpc
import numpy as np
from casadi import *
from casadi.tools import *
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
import pdb
import sys
import globals

sys.path.append('../../')

# Colors
PREDICTION = '#BA4A00'
CAR = '#F1C40F'
CAR_OUTLINE = '#B7950B'


class MPC:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.model = vehicle.model

        self.horizon = 10
        globals.horizon = self.horizon  # for model.py use

        self.Ts = 0.05
        self.length = 0.12
        self.width = 0.06

        self.current_prediction = None

        self.mpc = do_mpc.controller.MPC(self.model)
        setup_mpc = {
            'n_robust': 0,
            'n_horizon': self.horizon,
            't_step': self.Ts,
            'state_discretization': 'collocation',
            'store_full_solution': True,
        }
        self.mpc.set_param(**setup_mpc)

        # define the objective function and constriants
        self.objective_function_setup()
        self.constraints_setup()

        # provide time-varing parameters: setpoints/references
        self.tvp_template = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(self.tvp_fun)

        self.mpc.setup()

    def tvp_fun(self, t_now):
        for k in range(self.horizon):
            # extract information from current waypoint
            current_waypoint = self.vehicle.reference_path.get_waypoint(
                self.vehicle.wp_id + k
            )
            self.tvp_template['_tvp', k, 'x_ref'] = current_waypoint.x
            self.tvp_template['_tvp', k, 'y_ref'] = current_waypoint.y
            self.tvp_template['_tvp', k, 'psi_ref'] = current_waypoint.psi
            if current_waypoint.v_ref is not None:
                self.tvp_template['_tvp', k, 'vel_ref'] = current_waypoint.v_ref
            else:
                self.tvp_template['_tvp', k, 'vel_ref'] = 0

        return self.tvp_template

    def objective_function_setup(self):
        lterm = (
            self.model.aux['e_y'] ** 2
            + self.model.aux['psi_diff'] ** 2 
            + 10 * (self.model.x['pos_x'] - self.model.tvp['x_ref']) ** 2 
            + 10 * (self.model.x['pos_y'] - self.model.tvp['y_ref']) ** 2)

        mterm = (
            self.model.x['pos_x'] - self.model.tvp['x_ref'] ** 2 
            + (self.model.x['pos_y'] - self.model.tvp['y_ref']) ** 2 
            + (self.model.x['vel'] - self.model.tvp['vel_ref']) ** 2)

        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        self.mpc.set_rterm(acc=0.01, delta=0.01)

    def constraints_setup(
        self, vel_bound=[0.0, 1.0], e_y_bound=[0.0, 1.0], reset=False
    ):

        # states constraints
        self.mpc.bounds['lower', '_x', 'pos_x'] = -np.inf
        self.mpc.bounds['upper', '_x', 'pos_x'] = np.inf
        self.mpc.bounds['lower', '_x', 'pos_y'] = -np.inf
        self.mpc.bounds['upper', '_x', 'pos_y'] = np.inf
        self.mpc.bounds['lower', '_x', 'psi'] = - 2 * np.pi
        self.mpc.bounds['upper', '_x', 'psi'] = 2 * np.pi
        self.mpc.bounds['lower', '_x', 'vel'] = vel_bound[0]
        self.mpc.bounds['upper', '_x', 'vel'] = vel_bound[1]
        # self.mpc.bounds['lower', '_x', 'e_y'] = e_y_bound[0]
        # self.mpc.bounds['upper', '_x', 'e_y'] = e_y_bound[1]
        # self.mpc.bounds['lower', '_x', 'e_psi'] = -2 * np.pi
        # self.mpc.bounds['upper', '_x', 'e_psi'] = 2 * np.pi

        # input constraints
        delta_max = 0.66

        self.mpc.bounds['lower', '_u', 'acc'] = -0.5
        self.mpc.bounds['upper', '_u', 'acc'] = 0.5
        # self.mpc.bounds['lower', '_u', 'delta'] = - np.tan(delta_max) / self.length
        # self.mpc.bounds['upper', '_u', 'delta'] = np.tan(delta_max) / self.length
        self.mpc.bounds['lower', '_u', 'delta'] = -0.8
        self.mpc.bounds['upper', '_u', 'delta'] = 0.8

        if reset is True:
            self.mpc.setup()

    def get_control(self, x0):

        # update current waypoint
        self.vehicle.get_current_waypoint()

        # solve optization problem
        u0 = self.mpc.make_step(x0)

        # update predicted states
        # TODO: this is for plotting predicted trajectory
        #       but this solver doesn't reveal inner states
        #       but only final control signal
        # self.current_prediction = self.update_prediction()

        return np.array([u0[0], u0[1]])

    def distance_update(self, states):
        vel, psi = states[3], states[2]

        # Compute velocity along path
        # TODO: need to confirm the equation
        s_dot = vel * np.cos(self.mpc.data['_aux', 'psi_diff'][0])

        # Update distance travelled along reference path
        globals.s += s_dot * self.Ts

    def update_prediction(self):
        '''
        Transform the predicted states to predicted x and y coordinates.
        Mainly for visualization purposes.
        :param spatial_state_prediction: list of predicted state variables
        :return: lists of predicted x and y coordinates
        '''

        # Containers for x and y coordinates of predicted states
        x_pred, y_pred = [], []

        # Iterate over prediction horizon
        for n in range(2, self.horizon):
            # Get associated waypoint
            waypoint = self.vehicle.reference_path.get_waypoint(self.vehicle.wp_id + n)

            # Save predicted coordinates in world coordinate frame
            x_pred.append(waypoint.x)
            y_pred.append(waypoint.y)

        return x_pred, y_pred

    def show_prediction(self):
        '''
        Display predicted car trajectory in current axis.
        '''

        if self.current_prediction is not None:
            plt.scatter(
                self.current_prediction[0],
                self.current_prediction[1],
                c=PREDICTION,
                s=30,
            )
