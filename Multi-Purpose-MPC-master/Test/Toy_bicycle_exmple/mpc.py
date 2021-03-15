import do_mpc
import numpy as np
from casadi import *
from casadi.tools import *
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
import pdb
import sys
sys.path.append("../../")

# Colors
PREDICTION = '#BA4A00'
CAR = '#F1C40F'
CAR_OUTLINE = '#B7950B'


class MPC:
    def __init__(self, vehicle):

        self.vehicle = vehicle
        self.model = vehicle.model

        self.horizon = 30

        self.Ts = 0.05
        self.length = 0.12
        self.width = 0.06

        self.mpc = do_mpc.controller.MPC(self.model)
        # 'n_robust': 1,
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

        x = np.array([5 for i in range(self.horizon+1)])
        y = np.array([5 for i in range(self.horizon+1)])

        for k in range(self.horizon+1):
            self.tvp_template['_tvp', k, 'ref_x'] = x[k]
            self.tvp_template['_tvp', k, 'ref_y'] = y[k]

        return self.tvp_template

    def objective_function_setup(self):
        lterm = (
            (self.model.x['pos_x'] - self.model.tvp['ref_x']) ** 2
            + (self.model.x['pos_y'] - self.model.tvp['ref_y']) ** 2
            + self.model.x['e_y'] ** 2
            + self.model.x['e_psi'] ** 2
            + (self.model.x['vel'] - 1) ** 2
        )

        mterm = lterm

        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        self.mpc.set_rterm(acc=1e2, delta=1e2)

    def constraints_setup(self, vel_bound=[0.0, 1.0], e_y_bound=[0.0, 1.0], reset=False):

        # states constraints
        self.mpc.bounds['lower', '_x', 'pos_x'] = -np.inf
        self.mpc.bounds['upper', '_x', 'pos_x'] = np.inf
        self.mpc.bounds['lower', '_x', 'pos_y'] = -np.inf
        self.mpc.bounds['upper', '_x', 'pos_y'] = np.inf
        self.mpc.bounds['lower', '_x', 'psi'] = -2*np.pi
        self.mpc.bounds['upper', '_x', 'psi'] = 2*np.pi
        self.mpc.bounds['lower', '_x', 'vel'] = -5
        self.mpc.bounds['upper', '_x', 'vel'] = 5
        self.mpc.bounds['lower', '_x', 'e_y'] = -50
        self.mpc.bounds['upper', '_x', 'e_y'] = 50
        self.mpc.bounds['lower', '_x', 'e_psi'] = -2*np.pi
        self.mpc.bounds['upper', '_x', 'e_psi'] = 2*np.pi

        # input constraints
        delta_max = 0.66

        self.mpc.bounds['lower', '_u', 'acc'] = -1
        self.mpc.bounds['upper', '_u', 'acc'] = 1
        self.mpc.bounds['lower', '_u', 'delta'] = - \
            np.tan(delta_max) / self.length
        self.mpc.bounds['upper', '_u', 'delta'] = np.tan(
            delta_max) / self.length

        if reset is True:
            self.mpc.setup()

    def get_control(self, x0):

        # solve optization problem
        u0 = self.mpc.make_step(x0)

        return np.array([u0[0], u0[1]])
