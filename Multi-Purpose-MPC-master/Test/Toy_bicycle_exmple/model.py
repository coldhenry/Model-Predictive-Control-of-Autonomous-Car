import do_mpc
import numpy as np
from casadi import *
from casadi.tools import *
import matplotlib.pyplot as plt
import math
import pdb
import sys

sys.path.append("../../")


class simple_bycicle_model:
    def __init__(self, length, width, Ts):

        # car paramters
        self.length = length
        self.width = width

        # model
        self.Ts = Ts
        self.model = None

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
        e_psi = self.model.set_variable(var_type='_x', var_name='e_psi')

        # Input struct (optimization variables):
        acc = self.model.set_variable(var_type='_u', var_name='acc')
        delta = self.model.set_variable(var_type='_u', var_name='delta')

        # setpoint
        ref_x = self.model.set_variable(var_type='_tvp', var_name='ref_x')
        ref_y = self.model.set_variable(var_type='_tvp', var_name='ref_y')

        self.model.set_rhs('pos_x', vel * cos(psi))
        self.model.set_rhs('pos_y', vel * sin(psi))
        self.model.set_rhs('psi', vel / self.length * tan(delta))
        self.model.set_rhs('vel', acc)
        self.model.set_rhs('e_y', vel * sin(e_y))
        self.model.set_rhs('e_psi', vel / self.length * tan(delta))

        self.model.setup()
