import do_mpc
import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')


def MPC(model, ref):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_robust': 0,
        'n_horizon': 7,
        't_step': 0.5,
        'state_discretization': 'discrete',
        'store_full_solution': True,
    }

    mpc.set_param(**setup_mpc)

    mterm = (model.x['e_y'] - ref['e_y'])**2 + \
        (model.x['e_psi'] - ref['e_psi'])**2
    lterm = (model.x['e_y'] - ref['e_y'])**2 + \
        (model.x['e_psi'] - ref['e_psi'])**2

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=1e-4)

    mpc.bounds['lower', '_x', 'pos_x'] = -np.inf
    mpc.bounds['upper', '_x', 'pos_x'] = np.inf
    mpc.bounds['lower', '_x', 'pos_y'] = -np.inf
    mpc.bounds['upper', '_x', 'pos_y'] = np.inf
    mpc.bounds['lower', '_x', 'psi'] = -np.inf
    mpc.bounds['upper', '_x', 'psi'] = np.inf
    mpc.bounds['lower', '_x', 'vel'] = -3
    mpc.bounds['upper', '_x', 'vel'] = 3
    mpc.bounds['lower', '_x', 'e_y'] = -5
    mpc.bounds['upper', '_x', 'e_y'] = 5
    mpc.bounds['lower', '_x', 'e_psi'] = -6
    mpc.bounds['upper', '_x', 'e_psi'] = 6

    mpc.bounds['lower', '_u', 'acc'] = -1
    mpc.bounds['upper', '_u', 'acc'] = 1
    mpc.bounds['lower', '_u', 'delta'] = -0.5
    mpc.bounds['upper', '_u', 'delta'] = 0.5

    mpc.setup()

    return mpc
