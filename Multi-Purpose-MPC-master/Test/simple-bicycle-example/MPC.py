import do_mpc
import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys

sys.path.append("../../")


class MPC():

    def __init__(self, model):

        self.model = model
        self.mpc = do_mpc.controller.MPC(model)
        setup_mpc = {
            "n_robust": 0,
            "n_horizon": 7,
            "t_step": 0.05,
            "state_discretization": "discrete",
            "store_full_solution": True,
        }
        self.mpc.set_param(**setup_mpc)

        self.tvp_template = self.mpc.get_tvp_template()

        self.objective_function_setup()

        self.constraints_setup()

        self.mpc.setup()

    def tvp_fun(self, t_now):
        pass

    def objective_function_setup(self):
        lterm = (
            (self.model.x["e_y"] - self.model.tvp["e_y_ref"]) ** 2
            + (self.model.x["e_psi"] - self.model.tvp["e_psi_ref"]) ** 2
            + (self.model.x["vel"] - self.model.tvp["vel_ref"]) ** 2
            + self.model.x["delta"] ** 2
            + self.model.x["acc"] ** 2
        )
        mterm = lterm

        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        self.mpc.set_rterm(u=1e-4)

    def constraints_setup(self):
        # states constraints
        self.mpc.bounds["lower", "_x", "pos_x"] = -np.inf
        self.mpc.bounds["upper", "_x", "pos_x"] = np.inf
        self.mpc.bounds["lower", "_x", "pos_y"] = -np.inf
        self.mpc.bounds["upper", "_x", "pos_y"] = np.inf
        self.mpc.bounds["lower", "_x", "psi"] = -np.inf
        self.mpc.bounds["upper", "_x", "psi"] = np.inf
        self.mpc.bounds["lower", "_x", "vel"] = -3
        self.mpc.bounds["upper", "_x", "vel"] = 3
        self.mpc.bounds["lower", "_x", "e_y"] = -5
        self.mpc.bounds["upper", "_x", "e_y"] = 5
        self.mpc.bounds["lower", "_x", "e_psi"] = -6
        self.mpc.bounds["upper", "_x", "e_psi"] = 6

        # input constraints
        self.mpc.bounds["lower", "_u", "acc"] = -1
        self.mpc.bounds["upper", "_u", "acc"] = 1
        self.mpc.bounds["lower", "_u", "delta"] = -0.5
        self.mpc.bounds["upper", "_u", "delta"] = 0.5
