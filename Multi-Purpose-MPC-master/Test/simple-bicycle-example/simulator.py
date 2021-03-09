import do_mpc
import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')


def Simulator(model):

    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step=0.5)

    simulator.setup()

    return simulator
