
import numpy as np
from ase.calculators.calculator import Calculator
import tensorflow as tf

class ss_calc(Calculator, object):
    """
    Young-Jae Choi
    Physics Dep. POSTECH, south Korea
    """
    implemented_properties = ['energy', 'forces', 'atomic_energies']

    def __init__(
        self,
        model,
        var_ckpt,
        label='ss_calc',
        atoms=None,
        log_f='log.txt',
        ):
        """
        model (str or object)
            - (str)    Path to the saved model in pickle format.
            - (object) Object of one of these classes: (NN_force, ). Must have same training parameters used to make the var_ckpt file.
        var_ckpt (str) - Tensorflow variables checkpoint file.
        """

        # Initialize
        Calculator.__init__(self, label=label, atoms=atoms)

        ## Global variables
        # Logger
        from ss_util import Logger

        # Model
        if isinstance(model, str):
            import pickle as pckl
            with open(model, 'rb') as _:
                self.model = pckl.load(_)
        else:
            self.model = model
        self.model.log = Logger(log_f)
        self.model.dscrptr.log = model.log

        # Build an NN
        self.X, self.X_deriv, self.Neigh_ind, self.OL, F_hat, E_hat, self.DR = model.build_nn()
        # self.E = self.get_energy_tensor()
        self.F = self.get_force_tensor()

        # Load the saved variables
        self.sess = tf.Session()
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, var_ckpt)

    # def get_energy_tensor(self):
        # E = tf.reduce_sum(tf.concat(
            # self.OL,
            # axis=0,
            # ))
        # return E

    def get_force_tensor(self):
        """
        """
        # Reorder Neigh_ind
        #--> shape of (len_atoms, num_cutoff)
        Neigh_ind = tf.concat(self.Neigh_ind, axis=0)

        # Calc F_ij
        F_ij = []
        for spec in self.model.dscrptr.type_unique:
            F_ij.append(tf.reshape(tf.matmul(
                tf.reshape(self.X_deriv[spec], [-1, 3, self.model.dscrptr.len_dscrptr]),
                tf.reshape(tf.gradients(self.OL[spec], self.X[spec])[0], [-1, self.model.dscrptr.len_dscrptr, 1]),
                ), [-1, self.model.dscrptr.num_cutoff, 3]))
        #--> shape of (len_atoms, num_cutoff, 3)
        F_ij = tf.concat(F_ij, axis=0)

        # First (self) term of forces.
        #--> shape of (num_batch, len_atoms, 3)
        f_self = -tf.reduce_sum(F_ij, axis=1)

        # Second (cross) term of forces.
        len_atoms = tf.shape(F_ij)[0]
        a_bool = tf.equal(
                    #--> shape of (len_atoms, len_atoms, num_cutoff)
                    tf.tile(
                        tf.expand_dims(Neigh_ind, axis=0),
                        [len_atoms,1,1],
                        ),
                    tf.reshape(tf.range(len_atoms), [len_atoms,1,1]),
                    )

                # --> shape of (len_atoms, 3)
        f_cross = tf.reduce_sum(
            #--> shape of (len_atoms, len_atoms, 3)
            tf.reduce_sum(
                tf.ragged.boolean_mask(
                    #--> shape of (len_atoms, len_atoms, num_cutoff, 3)
                    tf.tile(
                        tf.expand_dims(
                            F_ij,
                            axis=0,
                            ),
                        [len_atoms,1,1,1],
                        ),
                    #--> shape of (len_atoms, len_atoms, num_cutoff)
                    a_bool,
                    name='neigh_mask',
                    ),
                axis=2,
                ),
            axis=1,
            )

        return tf.reshape(f_self + f_cross, [-1, 3])

    def calculate(
        self,
        atoms,
        properties,
        system_changes,
        ):
        """
        Shape of fgpt: (len(image) (==1 here), len(atoms), np.sum(num_cutoff (is a parameter in dscrptr class)), len(dscrptr))
        """

        # Load ASE's calculator class
        Calculator.calculate(self, atoms, properties, system_changes)

        # Get fgpt and fgpt derivative.
        fgpt, fgpt_deriv, neigh_ind, types, types_chem = self.model.dscrptr.gen_fgpts([atoms]) 

        # You have only one image here. Remove outmost shell.
        fgpt       = fgpt[0]
        fgpt_deriv = fgpt_deriv[0]
        neigh_ind  = neigh_ind[0]

        # Reshape.
        (len_atoms, num_cutoff, len_dscrptr) = fgpt.shape

        len_fgpt   = num_cutoff * len_dscrptr
        fgpt       = fgpt.reshape([len_atoms, len_fgpt])
        fgpt_deriv = fgpt_deriv.reshape([len_atoms, num_cutoff, 3, len_dscrptr])

        # Defind some variables.
        type_unique, type_count = np.unique(types, return_counts=True)

        # Calculate energy and forces.
        (atomic_energies_not_in_order, forces_not_in_order) = self.sess.run(
            (self.OL, self.F),
            feed_dict={A:B for A,B in list(zip(self.X,         [fgpt[types == spec] for spec in type_unique])) \
                                    + list(zip(self.X_deriv,   [fgpt_deriv[types == spec] for spec in type_unique])) \
                                    + list(zip(self.Neigh_ind, [neigh_ind[types == spec] for spec in type_unique])) \
                                    +     [   [self.DR,        1.0]]},
            )

        # Concate wrt species.
        atomic_energies_not_in_order = np.concatenate(atomic_energies_not_in_order, axis=0)

        # Make it in right order.
        atomic_energies = atomic_energies_not_in_order.copy()
        forces          = forces_not_in_order.copy()

        start_point = 0
        for spec in type_unique:
            atomic_energies[types == spec] = atomic_energies_not_in_order[start_point : start_point + type_count[spec]]
            forces         [types == spec] = forces_not_in_order         [start_point : start_point + type_count[spec]]
            start_point += type_count[spec]

        # Save results
        self.results['atomic_energies'] = atomic_energies
        self.results['energy']          = float(np.sum(atomic_energies))
        self.results['forces']          = forces
