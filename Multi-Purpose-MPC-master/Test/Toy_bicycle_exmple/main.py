from simulator import Simulator
from mpc import MPC
from model import simple_bycicle_model
import do_mpc
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import time
sys.path.append('../../')


""" User settings: """
show_animation = True
store_results = False

"""
Get configured do-mpc modules:
"""
vehicle = simple_bycicle_model(length=0.12, width=0.06, Ts=0.05)
model = vehicle.model

controller = MPC(vehicle)
mpc = controller.mpc

simulator = Simulator(vehicle).simulator


"""
Set initial state
"""

x0 = x0 = np.array([0, 0, 0, 0, 0, 0])
mpc.x0 = x0
simulator.x0 = x0

# Use initial state to set the initial guess.
mpc.set_initial_guess()

"""
Setup graphic:
"""

fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data)
plt.ion()

"""
Run MPC main loop:
"""


for k in range(150):
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)

    if show_animation:
        graphics.plot_results(t_ind=k)
        graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'lateral control')
