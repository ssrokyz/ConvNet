import tensorflow as tf
from ss_util import Logger, nospace
import numpy as np
import subprocess as sp
import datetime
import time
import pickle as pckl

def identity(x): return x

def plain_layer(
    layer_name,
    act_ftn,
    PL,
    node_num,
    b_std,
    dropout_rate,
    dtype,
    ):
    PL_node_num = PL.get_shape().as_list()[1]
    W      = tf.get_variable(layer_name+'_W', (PL_node_num, node_num), dtype, tf.random_uniform_initializer(0., np.sqrt(6./(PL_node_num + node_num), dtype=dtype), dtype=dtype))
    # W      = tf.get_variable(layer_name+'_W', (PL_node_num, node_num), dtype, tf.random_normal_initializer(0., 6./(PL_node_num + node_num), dtype=dtype))
    B      = tf.get_variable(layer_name+'_B', (node_num),              dtype, tf.random_normal_initializer(0., b_std,                       dtype=dtype))
    HL     = tf.nn.dropout(act_ftn(PL @ W + B), dropout_rate)
    return W, B, HL
    
def RN_layer(
    layer_name,
    act_ftn,
    PL,
    node_num,
    SL,
    b_std,
    dropout_rate,
    dtype,
    ):
    """
    layer_name      - Just layer name.
    act_ftn         - Activation function
    PL              - Previous layer
    node_num        - Number of nodes for this layer
    SL              - Skipped layer
    b_std           - Standard deviation for bais value distribution
    dropout_rate    - Dropout rate
    dtype           - Data type
    """
    PL_node_num = PL.get_shape().as_list()[1]
    W      = tf.get_variable(layer_name+'_W', (PL_node_num, node_num), dtype, tf.random_uniform_initializer(0., np.sqrt(6./(PL_node_num + node_num), dtype=dtype), dtype=dtype))
    # W      = tf.get_variable(layer_name+'_W', (PL_node_num, node_num), dtype, tf.random_normal_initializer(0., 6./(PL_node_num + node_num), dtype=dtype))
    B      = tf.get_variable(layer_name+'_B', (node_num),              dtype, tf.random_normal_initializer(0., b_std,                       dtype=dtype))
    HL     = tf.nn.dropout(act_ftn(PL @ W + B + SL), dropout_rate)
    return W, B, HL

def conv_layer_1d(
    act_ftn,
    PL,
    len_filter,
    num_channel,
    pooling_bool,
    dropout_rate,
    ):
    """
    In conv layer, len_filter means side length of the 2-D convolution filter.
    """
    HL = tf.layers.conv1d(PL, num_channel, len_filter, activation=act_ftn)
    if pooling_bool:
        HL = tf.layers.average_pooling1d(HL, 2, 2, padding='VALID')
    HL = tf.layers.dropout(HL, dropout_rate)
    return HL

def conv_layer_2d(
    layer_name,
    act_ftn,
    PL,
    len_filter,
    num_channel,
    b_std,
    dropout_rate,
    dtype,
    ):
    """
    In conv layer, len_filter means side length of the 2-D convolution filter.
    """
    tmp_shape = PL.get_shape().as_list()
    PL_node_num = tmp_shape[1]
    node_num = PL_node_num - len_filter + 1 # No padding
    PL_num_channel = tmp_shape[3]
    #
    W = tf.get_variable(
        layer_name+'_W',
        (len_filter, len_filter, PL_num_channel, num_channel),
        dtype,
        tf.random_normal_initializer(0., 6./(PL_node_num + len_filter), dtype=dtype),
        )
    B = tf.get_variable(
        layer_name+'_B',
        (node_num, node_num, num_channel),
        dtype,
        tf.random_normal_initializer(0., b_std, dtype=dtype),
        )
    # HL = tf.nn.dropout(act_ftn(tf.nn.conv2d(PL, W, strides=[1,1,1,1], padding='VALID') + B), dropout_rate)
    HL = tf.nn.dropout(tf.nn.avg_pool(act_ftn(tf.nn.conv2d(PL,
                                                           W,
                                                           strides=[1,1,1,1],
                                                           padding='VALID',
                                                           ) + B),
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='VALID',
                                      ), dropout_rate) ## for pooling
    return W, B, HL

# def plain_train_nn(
    # name,
    # len_inp,
    # len_out,
    # hl_node_num_list,
    # act_ftn,
    # DR,
    # dtype,
    # ):

    # hl_n_max = np.amax(hl_node_num_list)
    # X = tf.placeholder(dtype, (None, len_inp))
    # #### 1st HL
    # W = [tf.Variable(tf.random_uniform([len_inp, hl_node_num_list[0]], -1./hl_n_max, 1./hl_n_max, dtype=dtype), dtype=dtype)]
    # B = [tf.Variable(tf.random_uniform([hl_node_num_list[0]], -1., 1., dtype=dtype), dtype=dtype)]
    # HL_tmp = act_ftn(tf.add(tf.matmul(X, W[0]), B[0]))
    # HL = [tf.nn.dropout(HL_tmp, DR)]
    # #### 2nd HL to last HL
    # for i in range(1,len(hl_node_num_list)):
        # W.append(tf.Variable(tf.random_uniform([hl_node_num_list[i-1], hl_node_num_list[i]], -1./hl_n_max, 1./hl_n_max, dtype=dtype), dtype=dtype))
        # B.append(tf.Variable(tf.random_uniform([hl_node_num_list[i]], -1., 1., dtype=dtype), dtype=dtype))
        # HL_tmp = act_ftn(tf.add(tf.matmul(HL[i-1], W[i]), B[i]))
        # HL.append(tf.nn.dropout(HL_tmp, DR))
    # #### OL
    # W.append(tf.Variable(tf.random_uniform([hl_node_num_list[-1], len_out], -1./hl_n_max, 1./hl_n_max, dtype=dtype), dtype=dtype))
    # B.append(tf.Variable(tf.random_uniform([len_out], -1., 1., dtype=dtype), dtype=dtype))
    # OL = tf.add(tf.matmul(HL[-1], W[-1]), B[-1])
    # # OL = act_ftn(tf.matmul(HL[-1], W[-1]) + B[-1])
    # return X, OL

def ResNet_train_nn(
    name,
    len_inp,
    len_out,
    hl_node_num_list,
    act_ftn,
    DR,
    dtype,
    ):

    b_std = 1e-1
    name = 'T'+str(name)
    W = []; B=[]; HL=[]
    X = tf.placeholder(dtype, (None,len_inp))
    #### 1st HL (or 2nd if ResNet)
    W_tmp, B_tmp, HL_tmp = plain_layer(
        name+'_HL0',
        act_ftn,
        X,
        hl_node_num_list[0],
        b_std,
        DR,
        dtype,
        )
    W.append(W_tmp); B.append(B_tmp); HL.append(HL_tmp)
    #### 2nd HL
    W_tmp, B_tmp, HL_tmp = plain_layer(
        name+'_HL1',
        act_ftn,
        HL[-1],
        hl_node_num_list[1],
        b_std,
        DR,
        dtype,
        )
    W.append(W_tmp); B.append(B_tmp); HL.append(HL_tmp)
    for i in range(1,int(len(hl_node_num_list)/2)):
        #### (2i+1)-th HL
        W_tmp, B_tmp, HL_tmp = RN_layer(
            name+'_HL'+str(2*i),
            act_ftn,
            HL[2*i-1],
            hl_node_num_list[2*i],
            HL[2*i-2],
            b_std,
            DR,
            dtype,
            )
        W.append(W_tmp); B.append(B_tmp); HL.append(HL_tmp)
        #### (2i+2)-th HL
        W_tmp, B_tmp, HL_tmp = plain_layer(
            name+'_HL'+str(2*i+1),
            act_ftn,
            HL[2*i],
            hl_node_num_list[2*i+1],
            b_std,
            DR,
            dtype,
            )
        W.append(W_tmp); B.append(B_tmp); HL.append(HL_tmp)
    #### OL
    W_tmp, B_tmp, OL = plain_layer(
        name+'_OL',
        identity,
        HL[-1],
        len_out,
        b_std,
        DR,
        dtype,
        )
    W.append(W_tmp); B.append(B_tmp)
    return X, OL

def conv_train_nn(
    name,
    len_inp,
    n_inp_channel,
    len_out,
    len_filter_list,
    num_channel_list,
    pooling_bool,
    act_ftn,
    DR,
    dtype,
    ):
    """
    len_filter_list[i] are the lengthes of the 1-d filters of the ConvNet.
    """
    #
    HL=[]
    X = tf.placeholder(dtype, (None,len_inp))
    # 1st Conv.
    HL_tmp = tf.reshape(X, [-1, int(len_inp / n_inp_channel), n_inp_channel])
    HL_tmp = conv_layer_1d(
        act_ftn,
        HL_tmp,
        len_filter_list[0],
        num_channel_list[0],
        pooling_bool,
        DR,
        )
    HL.append(HL_tmp)
    # 2nd ~ N-th --> Conv.
    for i in range(1,len(len_filter_list)):
        # (i+1)th Conv.
        HL_tmp = conv_layer_1d(
            act_ftn,
            HL[i-1],
            len_filter_list[i],
            num_channel_list[i],
            pooling_bool,
            DR,
            )
        HL.append(HL_tmp)
    # OL (Fully-connected layer again)
    OL = tf.contrib.layers.flatten(HL[-1])
    OL = tf.layers.dense(OL, len_out, activation=identity)
    return X, OL

class NN_force(object):
    def __init__(
        self,
        model,
        dscrptr,
        hl_node_num_list = None,
        len_filter_list  = None,
        num_channel_list = None,
        pooling_bool     = False,
        act_ftn          = tf.nn.tanh,
        load_ckpt        = None,
        log_f            = 'log.txt',
        ):
        """
        model (str)     = 'ResNet'/'ConvNet' are possible.
            if      model == 'ResNet', hl_node_num_list must be provided.
            else if model == 'ConvNet', len_filter_list and num_channel_list must be provided. pooling_bool is optional.
        load_ckpt (str) = path of ckpt to load.
        """
        ###### defining self params
        self.log_f            = log_f
        self.log              = Logger(log_f)
        self.dtype            = 'float64'
        self.npy_path         = 'npys'
        self.fgpt_path        = 'fgpts'
        self.save_path        = 'saved_ckpts'
        self.model            = model
        self.dscrptr          = dscrptr
        self.hl_node_num_list = hl_node_num_list
        self.len_filter_list  = len_filter_list
        self.num_channel_list = num_channel_list
        self.pooling_bool     = pooling_bool
        self.act_ftn          = act_ftn
        self.len_inp          = dscrptr.len_fgpt
        self.n_inp_channel    = dscrptr.len_dscrptr
        self.len_out          = 1

    def save_model(
        self,
        path,
        ):
        pckl_name = 'model.pckl'
        path_plus_name = path+'/'+pckl_name
        sp.call(['mkdir -p '+path], shell=True)
        with open(path_plus_name, 'wb') as f:
            del(self.log)
            del(self.dscrptr.log)
            pckl.dump(self, f)
        self.log         = Logger(self.log_f)
        self.dscrptr.log = Logger(self.log_f)
            

    # @classmethod
    # def load_model(
        # cls,
        # saved_pckl,
        # log_f = 'log.txt'
        # ):
        # with open(saved_pckl, 'rb') as f:
            # cls_obj = pckl.load(f)
        # cls_obj.log         = Logger(log_f)
        # cls_obj.dscrptr.log = Logger(log_f)
        # return cls_obj

    def get_E(
        self,
        OL,
        type_count,
        ):
        #--> shape of (num_batch)
        E = tf.reduce_sum(
            #--> shape of (num_batch, len_alist)
            tf.concat(
                [tf.reshape(OL[spec], [-1,type_count[spec]]) for spec in self.dscrptr.type_unique],
                axis=-1,
                ),
            axis=-1,
            )
        return E

    def get_E_RMSE(
        self,
        OL,
        E_hat,
        len_atoms,
        type_count,
        ):
        """
        Get energy per atom RMSE.
        """
        e_rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(
            self.get_E(
                OL,
                type_count,
                ),
            E_hat,
            )))) / len_atoms
        return e_rmse

    def get_F(
        self,
        X,
        X_deriv,
        Neigh_ind,
        OL,
        len_inp,
        len_atoms,
        type_count,
        ):

        # ## Original version
        # X_deriv2 = []
        # for spec in self.dscrptr.type_unique:
            # X_deriv2.append(tf.reshape(tf.transpose(tf.reshape(X_deriv[spec], [-1, self.dscrptr.num_cutoff, 3, self.n_inp_channel]), perm=[0,2,1,3]), [-1, 3, len_inp]))

        # F = -2. * tf.matmul(
            # tf.reshape(tf.concat(
                # X_deriv2,
                # axis=0,
                # ), [-1,3,len_inp]),
            # tf.reshape(tf.concat(
                # [tf.gradients(OL[spec], X[spec])[0] for spec in self.dscrptr.type_unique],
                # axis=0,
                # ), [-1,len_inp,1]),
            # )

        ## New version
        # Reorder Neigh_ind
        #--> shape of (num_batch, len_atoms, num_cutoff)
        Neigh_ind = tf.concat(
            [tf.reshape(Neigh_ind[spec], [-1, type_count[spec], self.dscrptr.num_cutoff]) for spec in self.dscrptr.type_unique],
            axis=1,
            )

        # Calc F_ij
        F_ij = []
        for spec in self.dscrptr.type_unique:
            F_ij.append(tf.reshape(tf.matmul(
                tf.reshape(X_deriv[spec], [-1, 3, self.n_inp_channel]),
                tf.reshape(tf.gradients(OL[spec], X[spec])[0], [-1, self.n_inp_channel, 1]),
                ), [-1, type_count[spec], self.dscrptr.num_cutoff, 3]))
        #--> shape of (num_batch, len_atoms, num_cutoff, 3)
        F_ij = tf.concat(F_ij, axis=1)

        # First (self) term of forces.
        #--> shape of (num_batch, len_atoms, 3)
        f_self = -tf.reduce_sum(F_ij, axis=2)

        # Second (cross) term of forces.

        a_bool = tf.tile(
            tf.expand_dims(
                tf.equal(
                    tf.tile(
                        tf.expand_dims(Neigh_ind, axis=1),
                        [1,len_atoms,1,1],
                        ),
                    tf.reshape(tf.range(len_atoms), [1,-1,1,1]),
                    ),
                axis=4,
                ),
            [1,1,1,1,3],
            )

        f_cross = tf.reduce_sum(
            tf.reshape(
                tf.where(
                    a_bool,
                    tf.tile(
                        tf.expand_dims(
                            F_ij,
                            axis=1,
                            ),
                        [1,len_atoms,1,1,1],
                        ),
                    tf.zeros_like(a_bool, dtype=F_ij.dtype),
                    ),
                [-1, len_atoms, len_atoms *self.dscrptr.num_cutoff, 3],
                ),
            axis= 2,
            )

        return tf.reshape(f_self + f_cross, [-1, 3])

        # #--> shape of (len_atoms, num_batch, 3)
        # f_cross = []
        # for a in range(len_atoms):
            # #--> shape of (num_batch, len_atoms, num_cutoff, 3)
            # a_bool = tf.tile(tf.expand_dims(tf.equal(Neigh_ind, a), axis=3), [1,1,1,3])

                        # #  --> shape of (num_batch, 3)
            # f_cross.append(tf.reduce_sum(
                # tf.reshape(
                    # # #--> shape of (num_batch, num_cutoff, num_cutoff, 3)
                    # # tf.batch_gather(
                        # # #--> shape of (num_batch, len_atoms, num_cutoff, 3)
                        # # tf.where(
                            # # a_bool,
                            # # F_ij,
                            # # tf.zeros_like(F_ij),
                            # # ),
                        # # #--> shape of (num_batch, num_cutoff)
                        # # Neigh_ind[:, a],
                        # # ),
                    # tf.where(
                        # a_bool,
                        # F_ij,
                        # tf.zeros_like(F_ij),
                        # ),
                    # [-1, len_atoms* self.dscrptr.num_cutoff, 3],
                    # ),
                # axis=1,
                # ))
           # # --> shape of (num_batch, len_atoms, 3) 
        # F += tf.transpose(f_cross, perm=[1,0,2])

        # return tf.reshape(F, [-1, 3])
    
    def get_F_RMSE(
        self,
        X,
        X_deriv,
        Neigh_ind,
        OL,
        F_hat,
        len_inp,
        len_atoms,
        type_count,
        ):

        # Reorder F_hat
        F_hat = tf.concat(
            [tf.reshape(F_hat[spec], [-1, type_count[spec], 3]) for spec in self.dscrptr.type_unique],
            axis=1,
            )

        # Get RMSE
        f_rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(
            tf.reshape(F_hat, [-1,3]),
            self.get_F(
                X,
                X_deriv,
                Neigh_ind,
                OL,
                len_inp,
                len_atoms,
                type_count,
                ),
            ))))
        return f_rmse

    def make_fgpts(
        self,
        ):
        ### load params
        dscrptr   = self.dscrptr
        npy_path  = self.npy_path
        fgpt_path = self.fgpt_path

        ### make fgpts
        sp.call('rm -rf old-{}'                       .format(fgpt_path)           , shell=True)
        sp.call('mkdir -p old-{}'                     .format(fgpt_path)           , shell=True)
        sp.call('mv {}/* old-{}'                      .format(fgpt_path, fgpt_path), shell=True)
        sp.call('mkdir -p {}/training/ {}/validation/'.format(fgpt_path, fgpt_path), shell=True)

        train_fgpts, train_fgpts_deriv, train_neigh_ind, train_rot_mat, types, types_chem = dscrptr.gen_fgpts(npy_path+'/training', rotational_variation=True)
        valid_fgpts, valid_fgpts_deriv, valid_neigh_ind, valid_rot_mat, types, types_chem = dscrptr.gen_fgpts(npy_path+'/validation', rotational_variation=True)

        np.save(fgpt_path+'/training/fgpts.npy'         , train_fgpts)
        np.save(fgpt_path+'/validation/fgpts.npy'       , valid_fgpts)
        np.save(fgpt_path+'/training/fgpts_deriv.npy'   , train_fgpts_deriv)
        np.save(fgpt_path+'/validation/fgpts_deriv.npy' , valid_fgpts_deriv)
        np.save(fgpt_path+'/training/neigh_ind.npy'     , train_neigh_ind)
        np.save(fgpt_path+'/validation/neigh_ind.npy'   , valid_neigh_ind)
        np.save(fgpt_path+'/training/rot_mat.npy'       , train_rot_mat)
        np.save(fgpt_path+'/validation/rot_mat.npy'     , valid_rot_mat)

        # fgpt-log (Not the training log)
        with open(fgpt_path+'/log.txt', 'w') as txt:
            txt.write('\ncutoff_radi     : '+str(dscrptr.cutoff_radi))
            txt.write('\nnum_cutoff      : '+str(dscrptr.num_cutoff))
            txt.write('\nmultipole_order : '+str(dscrptr.multipole_order))
            txt.write('\ntypes           : '+str(types))
            txt.write('\ntypes_chem      : '+str(types_chem))
        with open(fgpt_path+'/dscrptr.pckl', 'wb') as f:
            del(dscrptr.log)
            pckl.dump(dscrptr,    f)
            dscrptr.log = self.log
            pckl.dump(types,      f)
            pckl.dump(types_chem, f)
        return train_fgpts, valid_fgpts, train_fgpts_deriv, valid_fgpts_deriv, \
            train_neigh_ind, valid_neigh_ind, train_rot_mat, valid_rot_mat, types, types_chem

    def read_fgpts(
        self,
        ):
        npy_path  = self.npy_path
        fgpt_path = self.fgpt_path
        dtype     = self.dtype
        log       = self.log
        try:
            log('Checking fgpt npy files...', tic='read_fgpt', no_space=True)
            train_fgpts       = np.load(fgpt_path+'/training/fgpts.npy').astype(dtype)
            valid_fgpts       = np.load(fgpt_path+'/validation/fgpts.npy').astype(dtype)
            train_fgpts_deriv = np.load(fgpt_path+'/training/fgpts_deriv.npy').astype(dtype)
            valid_fgpts_deriv = np.load(fgpt_path+'/validation/fgpts_deriv.npy').astype(dtype)
            train_neigh_ind   = np.load(fgpt_path+'/training/neigh_ind.npy').astype(int)
            valid_neigh_ind   = np.load(fgpt_path+'/validation/neigh_ind.npy').astype(int)
            train_rot_mat     = np.load(fgpt_path+'/training/rot_mat.npy').astype(dtype)
            valid_rot_mat     = np.load(fgpt_path+'/validation/rot_mat.npy').astype(dtype)
            with open(fgpt_path+'/dscrptr.pckl', 'rb') as f:
                dscrptr     = pckl.load(f)
                dscrptr.log = log
                types       = pckl.load(f)
                types_chem  = pckl.load(f)
            log('Read fgpt npy files!', toc='read_fgpt', no_space=True)
            log('\n')
        except:
            log('xxxxxxxxxxxxxxxxx     No fgpts.npy files found.     xxxxxxxxxxxxxxxxxx'.center(120), no_space=True)
            log('\n')
        else:
            log('ooooooooooooooooo     fgpts.npy files found. Load from the file.     oooooooooooooooooo'.center(120), no_space=True)
            log('\n')
            log('==========================================     C A U T I O N     ========================================='.center(120))
            log('\n')
            log('Fgpt files you loaded might not be what you wanted. Please be careful with it.'.center(120), no_space=True)
            log('\n')
            log('=========================================================================================================='.center(120))
            log('\n')
            if not np.array_equal(dscrptr.cutoff_radi, self.dscrptr.cutoff_radi) or not np.array_equal(dscrptr.num_cutoff, self.dscrptr.num_cutoff) or not \
                    np.array_equal(dscrptr.multipole_order, self.dscrptr.multipole_order):
                log('xxxxxxxxxxxxxxxxxxxxx But it seems not consistent btw input options & saved fgpt files. Check and try again!!!! xxxxxxxxxxxxxxxxxxxx'.center(120))
                raise ValueError('xxxxxxxxxxxxxxxxxxxxx But it seems not consistent btw input options & saved fgpt files. Check and try again!!!! xxxxxxxxxxxxxxxxxxxx')
            log('\n')
            return train_fgpts, valid_fgpts, train_fgpts_deriv, valid_fgpts_deriv, train_neigh_ind, valid_neigh_ind, train_rot_mat, valid_rot_mat, types, types_chem

    def build_nn(
        self,
        ):
        dtype            = self.dtype
        dscrptr          = self.dscrptr
        hl_node_num_list = self.hl_node_num_list
        len_filter_list  = self.len_filter_list
        num_channel_list = self.num_channel_list
        pooling_bool     = self.pooling_bool
        act_ftn          = self.act_ftn
        len_inp          = self.len_inp
        n_inp_channel    = self.n_inp_channel
        len_out          = self.len_out

        X         = []
        X_deriv   = []
        Neigh_ind = []
        HL        = []
        OL        = []
        F         = []
        DR = tf.placeholder(dtype, name='DR')

        for spec in self.dscrptr.type_unique:
            if self.model == 'ResNet':
                X_tmp, OL_tmp = ResNet_train_nn(
                    spec,
                    len_inp,
                    len_out,
                    hl_node_num_list,
                    act_ftn,
                    DR,
                    dtype,
                    )
            elif self.model == 'ConvNet':
                X_tmp, OL_tmp = conv_train_nn(
                    spec,
                    len_inp,
                    n_inp_channel,
                    len_out,
                    len_filter_list,
                    num_channel_list,
                    pooling_bool,
                    act_ftn,
                    DR,
                    dtype,
                    )
            else:
                X_tmp, OL_tmp = plain_train_nn(
                    spec,
                    len_inp,
                    len_out,
                    hl_node_num_list,
                    act_ftn,
                    DR,
                    dtype,
                    )

            X        .append(X_tmp)
            OL       .append(OL_tmp)
            Neigh_ind.append(tf.placeholder('int32', name='Neigh_ind_'+str(spec)))
            X_deriv  .append(tf.placeholder(dtype, name='X_deriv_'+str(spec)))
            F        .append(tf.placeholder(dtype, name='F_'+str(spec)))
        E = tf.placeholder(dtype, name='E')

        return X, X_deriv, Neigh_ind, OL, F, E, DR

    def train_f(
        self,
        optimizer,
        start_lr,
        load_ckpt    = 'latest',
        batch_size   = 20,
        load_fgpts   = True,
        e_ratio      = 1.00,
        f_ratio      = 1e-2,
        regular_rate = 1e-3,
        dropout_rate = 1.00,
        lr_up_rate   = 0.75,
        lr_up_points = None,
        max_step     = 1000000,
        save_intvl   = 1000,
        log_intvl    = 1000,
        seed         = None,
        ):
        npy_path         = self.npy_path
        fgpt_path        = self.fgpt_path
        save_path        = self.save_path
        model            = self.model
        dscrptr          = self.dscrptr
        hl_node_num_list = self.hl_node_num_list
        len_filter_list  = self.len_filter_list
        num_channel_list = self.num_channel_list
        act_ftn          = self.act_ftn
        dtype            = self.dtype
        log              = self.log
        regular_rate     = np.array(regular_rate, dtype)

        ### log
        now = datetime.datetime.now()
        log('\n\n\n\n\n\n'+'================================================================================'.center(120)+'\n\n')
        log('Code developed by Young-Jae Choi of POSTECH of Korea'.center(120))
        log('ssrokyz@gmail.com'.center(120)+'\n')
        log('\n'+'================================================================================'.center(120)+'\n')
        log('Training process started'.center(120))
        time_now = now.strftime('%Y-%m-%d_%H:%M:%S')
        log(nospace('Training start time: '+time_now).center(120))
        log('\n'+'================================================================================'.center(120)+'\n')
        log('   __________________Global variables___(...Check carefully...)____________________________\n')
        log(' >>>> '+nospace('hl_node_num_list ='+str(hl_node_num_list)))
        log(' >>>> '+nospace('len_filter_list  ='+str(len_filter_list )))
        log(' >>>> '+nospace('num_channel_list ='+str(num_channel_list)))
        log(' >>>> '+nospace('act_Ftn          ='+str(act_ftn         )))
        log(' >>>> '+nospace('model            ='+str(model           )))
        log(' >>>> '+nospace('optimizer        ='+str(optimizer       )))
        log(' >>>> '+nospace('start_lr         ='+str(start_lr        )))
        log(' >>>> '+nospace('batch_size       ='+str(batch_size      )))
        log(' >>>> '+nospace('load_fgpts       ='+str(load_fgpts      )))
        log(' >>>> '+nospace('e_ratio          ='+str(e_ratio         )))
        log(' >>>> '+nospace('f_ratio          ='+str(f_ratio         )))
        log(' >>>> '+nospace('regular_rate     ='+str(regular_rate    )))
        log(' >>>> '+nospace('dropout_rate     ='+str(dropout_rate    )))
        log(' >>>> '+nospace('lr_up_rate       ='+str(lr_up_rate      )))
        log(' >>>> '+nospace('lr_up_points     ='+str(lr_up_points    )))
        log(' >>>> '+nospace('max_step         ='+str(max_step        )))
        log(' >>>> '+nospace('save_intvl       ='+str(save_intvl      )))
        log(' >>>> '+nospace('log_intvl        ='+str(log_intvl       )))
        log('   ________________________________________________________________________________________\n\n')

        if load_fgpts:
            try:
                train_fgpts, valid_fgpts, train_fgpts_deriv, valid_fgpts_deriv, train_neigh_ind, valid_neigh_ind, \
                train_rot_mat, valid_rot_mat, types, types_chem = self.read_fgpts()
            except:
                log('xxxxxxxxxxxxxxxxxxxx     Though you wanted to load fgpts. I faild.     xxxxxxxxxxxxxxxxxxxxxx'.center(120))
                load_fgpts = False
        if not load_fgpts:
            train_fgpts, valid_fgpts, train_fgpts_deriv, valid_fgpts_deriv, train_neigh_ind, valid_neigh_ind, \
            train_rot_mat, valid_rot_mat, types, types_chem = self.make_fgpts()
        #### types
        type_unique, type_count = np.unique(types, return_counts=True)
        ####
        t_fgpts_shape = train_fgpts.shape
        len_valid = len(valid_fgpts)
        len_atoms = t_fgpts_shape[1]
        len_inp = self.len_inp
        n_inp_channel = self.n_inp_channel
        len_out = self.len_out
        self.len_inp = len_inp
        self.n_inp_channel = n_inp_channel
        self.len_out = len_out

        ### fgpt log
        log('   _____________Fgpt global variables___(...Check carefully...)___________\n')
        log(' >>>> '+nospace('cutoff_radi     ='+str(dscrptr.cutoff_radi)))
        log(' >>>> '+nospace('num_cutoff      ='+str(dscrptr.num_cutoff)))
        log(' >>>> '+nospace('multipole_order ='+str(dscrptr.multipole_order)))
        log('   _______________________________________________________________________\n\n')
        log('   _____________Fgpt details____________(...Check carefully...)___________\n')
        log(' >>>> '+nospace('types         ='+str(types)))
        log(' >>>> '+nospace('types_chem    ='+str(types_chem)))
        log(' >>>> '+nospace('len_inp          ='+str(len_inp)))
        log(' >>>> '+nospace('n_inp_channel ='+str(n_inp_channel)))
        log(' >>>> '+nospace('len_out          ='+str(len_out)))
        log(' >>>> '+nospace('len_atoms          ='+str(len_atoms)))
        log('   _______________________________________________________________________\n\n')

        #####
        try:
            train_f = np.load(npy_path+'/training/force_rot.npy').astype(dtype)
            valid_f = np.load(npy_path+'/validation/force_rot.npy').astype(dtype)
            if not load_fgpts:
                log(nospace('force_rot.npy must be not consistant with fgpts. New calculation will be carried out.'))
                raise ValueError('force_rot.npy must be not consistant with fgpts. New calculation will be carried out.')
        except:
            log(nospace(' Failed to load rotated forces. Calculate rotation of Forces'))
            # from euler_rotation import vector_rotation as Vr
            train_f = np.squeeze(
                np.expand_dims(train_rot_mat, axis=1) @ \
                np.expand_dims(np.load(npy_path+'/training/force.npy').astype(dtype), axis=3),
                )
            valid_f = np.squeeze(
                np.expand_dims(valid_rot_mat, axis=1) @ \
                np.expand_dims(np.load(npy_path+'/validation/force.npy').astype(dtype), axis=3),
                )
            np.save(npy_path+'/training/force_rot.npy', train_f)
            np.save(npy_path+'/validation/force_rot.npy', valid_f)
        else:
            log(nospace(' Rotated forces loaded successfully'))
        ## Load energy data
        train_e = np.load(npy_path+'/training/energy.npy').astype(dtype)
        valid_e = np.load(npy_path+'/validation/energy.npy').astype(dtype)

        #### 
        log('    >>> Energy Shifting & Scaling <<<')
        log('   _____________________________________________________________________')
        # e_shift = np.mean(train_e)
        # log(' >>>>> '+nospace('enegy shifted by {:.4f}'.format(e_shift)))
        # train_e -= e_shift
        # valid_e -= e_shift
        # e_scale = np.std(train_e) / len_atoms / hl_node_num_list[-1] * 1e+3
        # log(' >>>>> '+nospace('energy scaled by {:.4f}'.format(e_scale)))
        # train_e /= e_scale
        # valid_e /= e_scale
        log(" >>>>> "+nospace("training   set's mean value of shifted & scaled energy : {:.4f}".format(np.mean(train_e))))
        log(" >>>>> "+nospace("validation set's mean value of shifted & scaled energy : {:.4f}".format(np.mean(valid_e))))
        log(" >>>>> "+nospace("training   set's std  value of shifted & scaled energy : {:.4f}".format(np.std (train_e))))
        log(" >>>>> "+nospace("validation set's std  value of shifted & scaled energy : {:.4f}".format(np.std (valid_e))))
        log('   _____________________________________________________________________\n\n')

        # Build model
        log('=============================================================================================='.center(120))
        log('>>>>>>>>>>>>>>>>>>>>>>>>>>>      Constructing the NN model       <<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.center(120))
        X, X_deriv, Neigh_ind, OL, F, E, DR = self.build_nn()
        log('>>>>>>>>>>>>>>>>>>>>>>>>>>>        Construction complete!        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.center(120))
        log('=============================================================================================='.center(120))

        # if self.w_list is not None:
            # X, X_deriv, HL, OL, OL_wo, F, W, B, E = self.load_nn (dropout_rate)
            # log('================================================================================================'.center(120))
            # log('>>>>>>>>>>>>>>>>>>>>>>>>>>>          Loading saved model           <<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.center(120))
            # log('================================================================================================'.center(120))
        # else:

        #### Loss functions & RMSEs
        e_rmse    = self.get_E_RMSE(OL, E, len_atoms, type_count)
        f_rmse    = self.get_F_RMSE(X, X_deriv, Neigh_ind, OL, F, len_inp, len_atoms, type_count)
        # loss    = tf.add_n([
            # e_ratio * e_rmse,
            # f_ratio * f_rmse,
            # tf.multiply(regular_rate, tf.add_n([tf.reduce_mean(tf.square(w)) for ww in W for w in ww])),
            # ])
        loss = e_ratio * e_rmse + f_ratio * f_rmse

        # learning rate
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learnrate = tf.Variable(start_lr, trainable=False)
        if lr_up_points is not None:
            lr_up_points = np.concatenate((lr_up_points, [999999999]))
            updatelearnrate = tf.assign(
                learnrate,
                tf.multiply(learnrate, lr_up_rate),
                ) 

        # Define optimizer
        # # v1.0
        # do_train = optimizer(learnrate).minimize(loss, global_step=global_step)
        # # v2.0
        # trainable_variables = tf.trainable_variables()
        # l_grad = opti.compute_gradients(loss)
        # do_train = opti.apply_gradients(l_grad)
        # v3.0
        opti = optimizer(learnrate)
        l_grad = opti.compute_gradients(loss)
        capped = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in l_grad]
        do_train = opti.apply_gradients(capped, global_step=global_step)

        #### tensorboard
        # tf.summary.scalar('Loss_Ftn', loss)
        # tf.summary.histogram('W1', W[0][0])
        # merged = tf.summary.merge_all()

        #### initialize or load
        sess = tf.Session()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
        
        if load_ckpt:
            log('Trying to load variables from checkpoint.')
            if load_ckpt == 'latest':
                ckpt = tf.train.get_checkpoint_state(save_path)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    load_ckpt = ckpt.model_checkpoint_path
                else:
                    load_ckpt = False
            elif isinstance(load_ckpt, str):
                if not tf.train.checkpoint_exists(load_ckpt):
                    load_ckpt = False
            if load_ckpt:
                log(load_ckpt)
                saver.restore(sess, load_ckpt)
                log('Variables successfully loaded!')
            else:
                log(' WARNING >> Failed to load checkpoint. Start from scratch. <<')
                load_ckpt = False
        if not load_ckpt:
            log('Initializing TF...', tic='init', no_space=True)
            sess.run(tf.global_variables_initializer())
            log('Initialization complete!', toc='init', no_space=True)
            # Save model
            log('Saving model...', tic='init', no_space=True)
            self.save_model(self.save_path)
            log('Model saved!', toc='init', no_space=True)

        #### tensorboard
        # writer = tf.summary.FileWriter('./board/sample_1', sess.graph)

        #### Batch
        num_batch         = int(len(train_fgpts)/batch_size)
        train_fgpts       = np.array(train_fgpts      [:batch_size*num_batch])
        train_fgpts_deriv = np.array(train_fgpts_deriv[:batch_size*num_batch])
        train_neigh_ind   = np.array(train_neigh_ind  [:batch_size*num_batch])
        train_f           = np.array(train_f          [:batch_size*num_batch])
        batch_e = np.array([train_e[b_i*batch_size:(b_i+1)*batch_size] for b_i in range(num_batch)])
        del(train_e)

        #### Devide batch by spec except for "energy". && Reshape.
        log('Devideing by spec...', tic='devide')

        # Training set
        batch_fgpt_spec = [np.reshape(
            train_fgpts[:, types == spec],
            [num_batch, batch_size*type_count[spec], len_inp],
            ) for spec in type_unique ]
        del(train_fgpts)                                          

        batch_fgpt_deriv_spec = [np.reshape(
            train_fgpts_deriv[:, types == spec],
            [num_batch, batch_size*type_count[spec], dscrptr.num_cutoff, 3, n_inp_channel],
            ) for spec in type_unique ]
        del(train_fgpts_deriv)                                    

        batch_neigh_ind_spec = [np.reshape(
            train_neigh_ind[:, types == spec],
            [num_batch, batch_size*type_count[spec], dscrptr.num_cutoff],
            ) for spec in type_unique ]
        del(train_neigh_ind)

        batch_f_spec = [np.reshape(
            train_f[:, types == spec],
            [num_batch, batch_size*type_count[spec], 3],
            ) for spec in type_unique ]
        del(train_f)                                              

        # Test set
        valid_fgpts_spec = [np.reshape(
            valid_fgpts[:, types == spec],
            [len_valid*type_count[spec], len_inp],
            ) for spec in type_unique ]
        del(valid_fgpts)                                          

        valid_fgpts_deriv_spec = [np.reshape(
            valid_fgpts_deriv[:, types == spec],
            [len_valid*type_count[spec], dscrptr.num_cutoff, 3, n_inp_channel],
            ) for spec in type_unique ]
        del(valid_fgpts_deriv)                                    

        valid_neigh_ind_spec = [np.reshape(
            valid_neigh_ind[:, types == spec],
            [len_valid*type_count[spec], dscrptr.num_cutoff],
            ) for spec in type_unique ]
        del(valid_neigh_ind)

        valid_f_spec = [np.reshape(
            valid_f[:, types == spec],
            [len_valid*type_count[spec], 3],
            ) for spec in type_unique ]
        del(valid_f)

        log('Devided by spec!!', toc='devide', no_space=True)

        #### Run the train
        t_loss_list = []
        lr_up_ind   = 0
        lr          = sess.run(learnrate)

        log('Now,,, Machine is running!!!    (lol) ^_^;; \n\n\n')
        log('  Step            Time          Train_loss  valid_loss  valid_loss   E/atom_RMSE  Force_RMSE  V_calc  learn_rate')
        log('_(iter)___________________________________________________w/o_DO________(eV)_______(eV/Ang)____(sec)______________')
        for step in range(max_step):
            #### batch evolve
            b_i = sess.run(global_step) % num_batch
            # b_i = 10
            batch_fgpts_i       = [batch_fgpt_spec      [spec][b_i] for spec in type_unique]
            batch_fgpts_deriv_i = [batch_fgpt_deriv_spec[spec][b_i] for spec in type_unique]
            batch_neigh_ind_i   = [batch_neigh_ind_spec [spec][b_i] for spec in type_unique]
            batch_e_i           = batch_e[b_i]
            batch_f_i           = [batch_f_spec         [spec][b_i] for spec in type_unique]
            if sess.run(global_step) % log_intvl == 0:
                tic = time.time()
                v_loss = sess.run(
                    loss,
                    feed_dict = {A:B for A,B in list(zip(X        , valid_fgpts_spec      )) \
                                              + list(zip(X_deriv  , valid_fgpts_deriv_spec)) \
                                              + list(zip(Neigh_ind, valid_neigh_ind_spec  )) \
                                              +     [   (E        , valid_e               )] \
                                              + list(zip(F        , valid_f_spec          )) \
                                              +     [   (DR       , dropout_rate          )]},
                    )
                v_wo_loss = sess.run(
                    loss,
                    feed_dict = {A:B for A,B in list(zip(X        , valid_fgpts_spec      )) \
                                              + list(zip(X_deriv  , valid_fgpts_deriv_spec)) \
                                              + list(zip(Neigh_ind, valid_neigh_ind_spec  )) \
                                              +     [   (E        , valid_e               )] \
                                              + list(zip(F        , valid_f_spec          )) \
                                              +     [   (DR       , 1.0                   )]},
                    )
                # # For test purpose
                # print(sess.run(
                    # X[0],
                    # feed_dict = {A:B for A,B in list(zip(X, valid_fgpts_spec)) \
                                              # },
                    # ).shape)
                # print(sess.run(
                    # OL[0],
                    # feed_dict = {A:B for A,B in list(zip(X, valid_fgpts_spec)) \
                                              # + [(DR, 1.0)]},
                    # ).shape)
                e_wo_rmse = sess.run(
                    e_rmse,
                    feed_dict = {A:B for A,B in list(zip(X , valid_fgpts_spec)) \
                                              +     [   (E , valid_e         )] \
                                              +     [   (DR, 1.0             )]},
                    )
                f_wo_rmse = sess.run(
                    f_rmse,
                    feed_dict = {A:B for A,B in list(zip(X        , valid_fgpts_spec      )) \
                                              + list(zip(X_deriv  , valid_fgpts_deriv_spec)) \
                                              + list(zip(Neigh_ind, valid_neigh_ind_spec  )) \
                                              +     [   (E        , valid_e               )] \
                                              + list(zip(F        , valid_f_spec          )) \
                                              +     [   (DR       , 1.0                   )]},
                    )
                toc = time.time()

                # Gather 
                t_loss_avg = np.mean(t_loss_list)

                # Write the log
                now = datetime.datetime.now()
                time_now = now.strftime('%Y-%m-%d_%H:%M:%S')
                log('{:7d}    '.format(sess.run(global_step)) \
                  + time_now+'  ' \
                  + '{:10.4e}  '.format(t_loss_avg) \
                  + '{:10.4e}  '.format(v_loss) \
                  + '{:10.4e}    '.format(v_wo_loss) \
                  + '{:10.4e}  '.format(e_wo_rmse) \
                  + '{:10.4e} '.format(f_wo_rmse) \
                  + '{:5d}  '.format(int(toc - tic)) \
                  + '{:10.4e}'.format(lr))
                t_loss_list = []

                ## Tensorboard
                # summary = sess.run(merged, feed_dict={A:B for A,B in
                    # list(zip(X, valid_fgpts_spec))+[(E, valid_e  )]+list(zip(F, valid_f_spec))+list(zip(X_deriv, valid_fgpts_deriv_spec))})
                # writer.add_summary(summary)

                # Sometimes diverge...
                if v_loss >= 1e+5 or str(v_loss) == 'nan':
                    log('Loss diverged. Please try again with smaller learning rate.')
                    raise ValueError('Loss diverged. Please try again with smaller learning rate.')

            # Update learning rate.
            if lr_up_points is not None:
                if sess.run(global_step) == lr_up_points[lr_up_ind]:
                    lr_up_ind += 1
                    lr = sess.run(updatelearnrate)

            # Save the checkpoint.
            if sess.run(global_step) % save_intvl == 0:
                tic = time.time()
                # self.w_list, self.b_list = sess.run([W, B])
                # self.save_model(save_path, step)
                saver.save(sess, './{}/{}.{:07d}'.format(self.save_path, model, sess.run(global_step)))
                toc = time.time()
                log(('>>> Variables saved! {} (sec) <<<'.format(int(toc - tic))).center(120), no_space=True)

            # Training step
            t_loss_tmp, _ = sess.run(
                    [loss, do_train],
                    feed_dict={A:B for A,B in list(zip(X        , batch_fgpts_i      )) \
                                            + list(zip(X_deriv  , batch_fgpts_deriv_i)) \
                                            + list(zip(Neigh_ind, batch_neigh_ind_i  )) \
                                            +     [   (E        , batch_e_i          )] \
                                            + list(zip(F        , batch_f_i          )) \
                                            +     [   (DR       , dropout_rate       )]},
                                            )

            # Save the training loss of a batch.
            t_loss_list.append(t_loss_tmp)
