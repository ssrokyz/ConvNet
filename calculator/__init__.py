
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
        ):
        """
        model (object) - Object of one of these classes: (NN_force, )
        var_ckpt (str) - Tensorflow variables checkpoint file.
        """

        # Initialize
        Calculator.__init__(self, label=label, atoms=atoms)

        # Global variables
        self.model = model

        # Build an NN
        self.X, self.X_deriv, HL, self.OL, F_hat, E_hat, DR = model.build_nn()
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
        F = 2. * tf.matmul(
            tf.reshape(tf.concat(
                self.X_deriv,
                axis=0,
                ), [-1,3,self.model.dscrptr.len_fgpt]),
            tf.reshape(tf.concat(
                [tf.gradients(self.OL[spec], self.X[spec])[0] for spec in self.model.dscrptr.type_unique],
                axis=0,
                ), [-1,self.model.dscrptr.len_fgpt,1]),
            )
        return tf.reshape(F, [-1, 3])

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
        fgpt, fgpt_deriv, types, types_chem = self.model.dscrptr.gen_fgpts([atoms]) 

        # You have only one image here. Remove outmost shell.
        fgpt       = fgpt[0]
        fgpt_deriv = fgpt_deriv[0]

        # Reshape.
        (len_atoms, num_cutoff, len_dscrptr) = fgpt.shape

        len_fgpt   = num_cutoff * len_dscrptr
        fgpt       = fgpt.reshape([len_atoms, len_fgpt])
        fgpt_deriv = fgpt_deriv.reshape([len_atoms, 3, len_fgpt])

        # Defind some variables.
        type_unique, type_count = np.unique(types, return_counts=True)

        # Calculate energy and forces.
        (atomic_energies_not_in_order, forces_not_in_order) = self.sess.run(
            (self.OL, self.F),
            feed_dict={A:B for A,B in list(zip(self.X, [fgpt[types == spec] for spec in type_unique])) \
                                    + list(zip(self.X_deriv, [fgpt_deriv[types == spec] for spec in type_unique]))},
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
        self.results['energy']          = np.sum(atomic_energies)
        self.results['forces']          = forces
