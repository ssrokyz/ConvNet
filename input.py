#!/usr/bin/env python

from dscrptr import vector
from tf_model import NN_force
import numpy as np
import tensorflow as tf
# np.set_printoptions(threshold=np.nan)

# NN = NN_force.load_model('saved_ckpts/25000.pckl')

log_file = 'log.txt'

try:
    NN is not None
except:
    dscrptr = vector(
        num_kind        = 3,
        cutoff_radi     = 4.5,
        num_cutoff      = 33,
        multipole_order = ['r', 2],
        logfile_name    = log_file,
        )
    # from ase.io import read
    # alist = read('test.traj', ':')
    # fgpt_vecs = dscrptr.gen_fgpts(
        # inp_type = 'alist',
        # path = alist,
        # # inp_type = 'npy',
        # # path = 'npys/validation',
        # )
    # print(str(fgpt_vecs.shape))

    NN = NN_force(
        'ConvNet',
        dscrptr,
        # hl_node_num_list = [25, 25, 25, 25],
        len_filter_list    = [  12,   6,   2],
        num_channel_list   = [  15,  30,  60],
        pooling_bool       = True,
        act_ftn  = 
        # tf.nn.relu,
        # tf.nn.tanh,
        tf.sin,
        # tf.nn.softplus,
        # tf.nn.elu,
        # tf.nn.crelu,
        # tf.nn.leaky_relu,
        # tf.nn.softmax,
        # tf.nn.softsign,
        # tf.nn.softsign,
        log_f    = log_file,
        )
NN.train_f(
    optimizer = 
        # tf.train.AdamOptimizer,
        tf.contrib.opt.NadamOptimizer,
        # tf.train.RMSPropOptimizer,
        # tf.train.GradientDescentOptimizer,
    start_lr     = 1e-3,
    load_ckpt    = 'latest',
    # load_ckpt    = False,
    batch_size   = 100,
    load_fgpts   = True,
    e_ratio      = 1e-0,
    f_ratio      = 1e-0,
    regular_rate = 0.00,
    dropout_rate = 1.00,
    lr_up_rate   = 0.10,
    # lr_up_points = [330000],
    lr_up_points = None,
    max_step     = 1000000,
    save_intvl   = 5000,
    log_intvl    = 1000,
    seed         = None,
    )


