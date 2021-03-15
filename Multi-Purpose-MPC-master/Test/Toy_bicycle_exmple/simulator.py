import do_mpc
import numpy as np
from casadi import *
from casadi.tools import *
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
import pdb
import sys
sys.path.append('../../')


class Simulator:

    def __init__(self, vehicle):

        self.vehicle = vehicle
        self.simulator = do_mpc.simulator.Simulator(vehicle.model)

        self.horizon = 30
        self.simulator.set_param(t_step=0.05)

        # provide time-varing parameters: setpoints/references
        self.tvp_template = self.simulator.get_tvp_template()
        self.simulator.set_tvp_fun(self.tvp_fun)

        self.simulator.setup()

    def tvp_fun(self, t_now):

        x = np.array([5 for i in range(self.horizon+1)])
        y = np.array([5 for i in range(self.horizon+1)])

        for k in range(self.horizon+1):
            self.tvp_template['ref_x'] = x[k]
            self.tvp_template['ref_y'] = y[k]

        return self.tvp_template
