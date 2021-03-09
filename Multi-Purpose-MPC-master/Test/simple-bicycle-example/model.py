import do_mpc
import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')


def simple_bycicle_model():

    model_type = 'discrete'  # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # States struct (optimization variables):
    pos_x = model.set_variable(var_type='_x', var_name='pos_x')
    pos_y = model.set_variable(var_type='_x', var_name='pos_y')
    psi = model.set_variable(var_type='_x', var_name='psi')
    vel = model.set_variable(var_type='_x', var_name='vel')
    e_y = model.set_variable(var_type='_x', var_name='e_y')
    e_psi = model.set_variable(var_type='_x', var_name='e_psi')

    # Input struct (optimization variables):
    acc = model.set_variable(var_type='_u', var_name='acc')
    delta = model.set_variable(var_type='_u', var_name='delta')

    # other parameters
    length = model.set_variable(var_type='_p', var_name='length')

    model.set_rhs('pos_x', vel * np.cos(psi))
    model.set_rhs('pos_y', vel * np.sin(psi))
    model.set_rhs('psi', vel/length * np.tan(delta))
    model.set_rhs('vel', acc)
    model.set_rhs('e_y', vel * np.sin(e_y))
    model.set_rhs('e_psi', vel/length * np.tan(delta))

    model.setup()


if __name__ == '__main__':
    simple_bycicle_model()
