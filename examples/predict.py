#!/usr/bin/env python

from __init__ import ss_calc
from dscrptr import vector
from tf_model import NN_force
import tensorflow as tf
from ase import units

#### Load NN model
import pickle as pckl
with open('../saved_ckpts/model.pckl', 'rb') as f:
    nn_obj = pckl.load(f)

# Load NN parameters
from ase.io import read, write
# alist = read('small.traj', ':')
calc = ss_calc(
    nn_obj,
    '../saved_ckpts/ConvNet.5105000',
    big_sys=False,
    )

#### Global params
## cell
label = "gst-crystallization"
atoms = read('init.traj')
##
temp     = 1000 *units.kB
d_t      = 10 *units.fs
t_step   = 5000
friction = 1e-3
##
atoms.set_calculator(calc)

#### Calculate
# new_alist = []
# import datetime
# for i in range(len(alist)):
    # atoms = alist[i]
    # calc.get_potential_energy(atoms)
    # results = calc.results.copy()
    # atoms._calc.results = results
    # atoms._calc.atoms = atoms
    # new_alist.append(atoms)
    # if i % 100 == 0:
        # now = datetime.datetime.now()
        # time = now.strftime('%Y-%m-%d %H:%M:%S')
        # print('step {} - time: {}'.format(i, time))
# write('predict.traj', new_alist)

#### dynamics
## Initialization
# Maxwell-Boltzmann distribution
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution as Max
Max(atoms, temp * 1)
# Multiply
# atoms.arrays['momenta'] *= temp / units.kB / atoms.get_temperature()
# Global
# Stationary(atoms)

## Dynamics
# from ase.md.verlet import VelocityVerlet
# dyn = VelocityVerlet(
    # atoms,
    # d_t,
    # trajectory = label+'.traj',
    # logfile = 'log_'+label+'.txt',
    # )
from ase.md import Langevin
dyn = Langevin(
    atoms       = atoms,
    timestep    = 10 *units.fs,
    temperature = temp,
    friction    = friction,
    trajectory  = label+'.traj',
    logfile     = 'log-'+label+'.txt',
    ) 
# from ase.md.npt import NPT
# dyn = NPT(
    # atoms = atoms,
    # timestep = d_t,
    # temperature = temp,
    # externalstress = 0.,
    # ttime = 75 * units.fs,
    # pfactor = (75. *units.fs)**2 * 100. *units.GPa,
    # trajectory  = label+'.traj',
    # logfile     = 'log_'+label+'.txt',
    # )
### relax option
# dyn.set_fraction_traceless(0) # 0 --> no shape change but yes volume change
dyn.run(steps=t_step)     #MD simulation of object 'dyn' is performed by 'run' method of VV class

