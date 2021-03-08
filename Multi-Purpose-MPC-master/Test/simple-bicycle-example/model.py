import do_mpc
import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')


def simple_bycicle_model():
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'discrete'  # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # States struct (optimization variables):
    _x = model.set_variable(var_type='_x', var_name='x', shape=(4, 1))

    # Input struct (optimization variables):
    _u = model.set_variable(var_type='_u', var_name='u', shape=(2, 1))

    # Set expression. These can be used in the cost function, as non-linear constraints
    model.set_expression(expr_name='cost', expr=sum1(_x**2))

    A = np.array([[0.763,  0.460,  0.115,  0.020],
                  [-0.899,  0.763,  0.420,  0.115],
                  [0.115,  0.020,  0.763,  0.460],
                  [0.420,  0.115, -0.899,  0.763]])

    B = np.array([[0.014],
                  [0.063],
                  [0.221],
                  [0.367]])

    x_next = A@_x+B@_u
    model.set_rhs('x', x_next)

    model.setup()

    return model
