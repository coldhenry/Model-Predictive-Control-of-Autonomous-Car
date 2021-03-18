import do_mpc
import numpy as np
from casadi import *
from casadi.tools import *
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
import pdb
import sys
import globals
sys.path.append("../../")

# Colors
PREDICTION = '#BA4A00'
CAR = '#F1C40F'
CAR_OUTLINE = '#B7950B'


class MPC:
    def __init__(self, vehicle):

        # vehicle information
        self.vehicle = vehicle
        self.model = vehicle.model
        self.length = 0.12
        self.width = 0.06

        # mpc configuration
        self.horizon = 15
        self.Ts = 0.05
        self.mpc = do_mpc.controller.MPC(self.model)

        # BUG a faster solver: but unable to use
        # self.mpc.set_param(nlpsol_opts={'ipopt.linear_solver': 'MA27'})

        setup_mpc = {
            'n_robust': 3,
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

        for k in range(self.horizon+1):
            # extract information from current waypoint
            current_waypoint = self.vehicle.reference_path.get_waypoint(
                self.vehicle.wp_id + k
            )
            print("cwp: {}, {}".format(current_waypoint.x, current_waypoint.y))
            self.tvp_template['_tvp', k, 'ref_x'] = current_waypoint.x
            self.tvp_template['_tvp', k, 'ref_y'] = current_waypoint.y
            self.tvp_template['_tvp', k, 'ref_psi'] = current_waypoint.psi

        return self.tvp_template

    def objective_function_setup(self):
        lterm = (
            10 * (self.model.x['pos_x'] - self.model.tvp['ref_x']) ** 2
            + 10 * (self.model.x['pos_y'] - self.model.tvp['ref_y']) ** 2
            + self.model.aux['e_psi_current'] ** 2
            + 5 * self.model.x['e_y'] ** 2
        )

        mterm = (
            (self.model.x['pos_x'] - self.model.tvp['ref_x']) ** 2
            + (self.model.x['pos_y'] - self.model.tvp['ref_y']) ** 2
        )

        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        # self.mpc.set_rterm(delta=1e-2)

    def constraints_setup(self, vel_bound=[0.0, 1.0], e_y_bound=[0.0, 1.0], reset=False):

        # states constraints
        self.mpc.bounds['lower', '_x', 'pos_x'] = -np.inf
        self.mpc.bounds['upper', '_x', 'pos_x'] = np.inf
        self.mpc.bounds['lower', '_x', 'pos_y'] = -np.inf
        self.mpc.bounds['upper', '_x', 'pos_y'] = np.inf
        self.mpc.bounds['lower', '_x', 'psi'] = - 2 * np.pi
        self.mpc.bounds['upper', '_x', 'psi'] = 2 * np.pi
        self.mpc.bounds['lower', '_x', 'vel'] = 0.0
        self.mpc.bounds['upper', '_x', 'vel'] = 0.5
        self.mpc.bounds['lower', '_x', 'e_y'] = - 0.5
        self.mpc.bounds['upper', '_x', 'e_y'] = 0.5

        # input constraints
        self.mpc.bounds['lower', '_u', 'acc'] = -0.1
        self.mpc.bounds['upper', '_u', 'acc'] = 0.5
        self.mpc.bounds['lower', '_u', 'delta'] = - 0.85
        self.mpc.bounds['upper', '_u', 'delta'] = 0.85

        if reset is True:
            self.mpc.setup()

    # def get_control(self, x0):

    #     # solve optization problem
    #     u0 = self.mpc.make_step(x0)

    #     return np.array([u0[0], u0[1]])

    def distance_update(self, x0):

        vel, e_psi = x0[3], self.mpc.data['_aux', 'e_psi_current'][0]

        # Compute velocity along path
        # TODO: need to confirm the equation
        # s_dot = vel * np.cos(e_psi)

        # Compute velocity along path
        s_dot = 1 / (1 - self.mpc.data['_x', 'e_y'][0] * self.vehicle.current_waypoint.kappa) \
            * vel * np.cos(e_psi)

        # Update distance travelled along reference path
        globals.s += s_dot * self.Ts
        print("traveled distance: ", globals.s)
