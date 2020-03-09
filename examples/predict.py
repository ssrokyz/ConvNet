#!/usr/bin/env python

from __init__ import ss_calc
from dscrptr import vector
from tf_model import NN_force
import tensorflow as tf

import pickle as pckl
with open('../saved_ckpts/model.pckl', 'rb') as f:
    nn_obj = pckl.load(f)

from ase.io import read, write
alist = read('small.traj', ':')
calc = ss_calc(nn_obj, '../saved_ckpts/ConvNet.0445000')
new_alist = []
import datetime
for i in range(len(alist)):
    atoms = alist[i]
    calc.get_potential_energy(atoms)
    results = calc.results.copy()
    atoms._calc.results = results
    atoms._calc.atoms = atoms
    new_alist.append(atoms)
    if i % 100 == 0:
        now = datetime.datetime.now()
        time = now.strftime('%Y-%m-%d %H:%M:%S')
        print('step {} - time: {}'.format(i, time))
write('predict.traj', new_alist)

