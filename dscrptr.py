import numpy as np
from ss_util import Logger, nospace

def get_new_axis(axis1, axis2, dtype='float32'):
    """
    Return three dimensional axis vectors with unit size.
    axis1 will be new x1 axis.
    """
    axis3 = np.cross(axis1, axis2)
    axis1 = np.array(axis1) / np.linalg.norm(axis1)
    axis3 = np.array(axis3) / np.linalg.norm(axis3)
    axis2 = np.cross(axis3, axis1)
    return np.array([axis1, axis2, axis3], dtype=dtype)

def read_npy(npy_path):
    """
    You have to make sure that npys have coordinate system "wrapped".
    """
    box            = np.load(npy_path+'/box.npy')
    coord          = np.load(npy_path+'/coord.npy')
    type_file      = open(npy_path+'/type.txt')
    types          = np.array(type_file.readline().split(), dtype = 'int32')
    types_chem     = type_file.readline().split()
    return box, coord, types, types_chem

def read_alist(alist):
    """
    Read atoms list and returns arrays
    """
    from traj2npy import alist2numpy as a2n
    box, coord, energy, force, stress = a2n(alist)
    
    # Check whether chems are consistent or not.
    types_chem = alist[0].get_chemical_symbols()
    for atoms in alist:
        if atoms.get_chemical_symbols() != types_chem:
            raise ValueError("Chemical symbols seem to be not consistent btw images. Please check")

    # Get integer-type 'types' array.
    from ss_util import list2numlist as l2nl
    types       = np.array(l2nl(list(types_chem)), dtype='int32')
    return box, coord, types, types_chem

def make_x2_supercell(
    box,
    coord,
    dtype='float32',
    ):
    """
    Make input cell as 2x2x2 supercell.
    box (arr)   = shape of (len_alist, 3, 3)
    coord (arr) = shape of (len_alist, len_atoms, 3)
    """
    # Make box
    x2_box   = []
    x2_coord = []
    for i in range(len(box)):
        x2_coord_i = []
        for G_z in range(2):
            for G_y in range(2):
                for G_x in range(2):
                    G = np.array([G_x, G_y, G_z], dtype='int32')
                    #                 --> shape of (len_atoms, 3)
                    x2_coord_i.extend(coord[i] + G)
        #               --> shape of (8*len_atoms, 3)
        x2_coord.append(np.array(x2_coord_i, dtype=dtype) / 2.)
        x2_box.append(box[i] * 2.)
    # Return shape of (len_alist, 8*len_atoms, 3)
    return np.array(x2_box, dtype=dtype), np.array(x2_coord, dtype=dtype)

def get_max_cutoff(
    box,
    ):
    """
    (Minimum value of inter-plane distance) / 2. == max cutoff length.
    """
    cutoff_list = []
    for i in range(len(box)):
        lattice = box[i]
        vol = np.linalg.det(lattice)
        d1 = vol / np.linalg.norm(np.cross(lattice[1],lattice[2]))
        d2 = vol / np.linalg.norm(np.cross(lattice[2],lattice[0]))
        d3 = vol / np.linalg.norm(np.cross(lattice[0],lattice[1]))
        cutoff_list.append(np.amin([d1, d2, d3]))
    return np.amin(cutoff_list) / 2.

class vector(object):
    """ 
    Sequential arrangement of invert radial-distances and unit vectors in accending order of radius.
    """
    def __init__(
        self,
        num_kind,
        num_cutoff,
        multipole_order = ['r', 2,],
        logfile_name    = 'log.txt',
        dtype           = 'float32',
        ):
        """

        num_kind (int)         = Number of species kinds in the composition.
        num_cutoff (int)       = Number of atoms used to make descriptors. It will be proportional to the length the of input layer.
        multipole_order (list) = List that specifies which descriptor basis you will use. If 'r' is in list, (1/r) will be used as a
                                 basis. If zero or positive integers are provided, (x_i/r^n) (x_i = {x,y,z}) will be used as basis

        """

        # Get descriptor length
        if 'r' in multipole_order:
            self.len_dscrptr = 3 * (len(multipole_order)-1) + 2
        else:
            self.len_dscrptr = 3 * len(multipole_order) + 1

        # defining self params
        self.num_cutoff      = np.array(num_cutoff, dtype = 'int32')
        self.multipole_order = list(multipole_order)
        self.type_unique     = list(range(num_kind))
        self.len_fgpt        = self.len_dscrptr * num_cutoff
        self.log             = Logger(logfile_name)
        self.dtype           = dtype

    def get_multipole_fgpt(
        self,
        r_vec_arr,
        r_arr,
        ):
        """
        Get fingerprint vector of dscrptr defined by 'self.multipole_order'.
        r_vec_arr (arr) = Array of vec(r) in shape (num_cutoff, 3).
        r_arr (arr)     = Array of |vec(r)| in shape (num_cutoff, 1). Note that the dimension is 2.
        """
        MO = self.multipole_order[:]
        fgpt = []
        if 'r' in self.multipole_order:
            MO.remove('r')
            fgpt += [1./r_arr]
        if MO:
            fgpt += [r_vec_arr / r_arr**n for n in MO]
        # Output is in shape of (num_cutoff, self.len_dscrptr -1)
        return np.concatenate(fgpt, axis=1)

    def get_multipole_fgpt_deriv(
        self,
        r_vec_arr,
        r_arr,
        ):
        """
        Get fingerprint derivative matrix of dscrptr defined by 'self.multipole_order'.
        r_vec_arr (arr) = Array of vec(r) in shape (num_cutoff, 3).
        r_arr (arr)     = Array of |vec(r)| in shape (num_cutoff, 1). Note that the dimension is 2.
        """
        MO = self.multipole_order[:]
        fgpt_deriv = []
        if 'r' in self.multipole_order:
            MO.remove('r')
            #             --> shape of (1, num_cutoff, 3, 1)
            #              --> shape of (num_cutoff, 3, 1)
            fgpt_deriv += [np.expand_dims(r_vec_arr / r_arr**3, axis=2)]
        if MO:
            #             --> shape of (len(MO) (-1 or not), num_cutoff, 3, 3)
            #              --> shape of (num_cutoff, 3, 3)
            fgpt_deriv += [-np.repeat(
                #           --> shape of (1, 3, 3)
                            [np.eye(3, dtype=self.dtype)],
                            len(r_arr),
                            axis=0,
                            #   --> shape of (num_cutoff, 1, 1)
                            ) / np.expand_dims(r_arr**n, axis=2) \
                            #   --> shape of (num_cutoff, 1, 1)
                              + np.expand_dims(n / r_arr**(n+2), axis=2) \
                            #   --> shape of (num_cutoff, 3, 3)
                              * np.matmul(
                              #   --> shape of (num_cutoff, 3, 1)
                                  np.expand_dims(r_vec_arr, axis=2),
                              #   --> shape of (num_cutoff, 1, 3)
                                  np.expand_dims(r_vec_arr, axis=1),
                                  ) \
                         for n in MO]
        # Output in shape of (num_cutoff, 3, len_dscrptr -1)
        return np.concatenate(fgpt_deriv, axis=2)

    def gen_fgpts(
        self,
        npy_path_or_list,
        rotational_variation=False,
        allow_supercell=False,
        ):
        """
        Generate fgpts and fgpt_derivs.
        npy_path_or_list (str/list) =
        rotational_variation (bool) = 
        """
        # make some values
        if isinstance(npy_path_or_list, list):
            box, coord, types, types_chem = read_alist(npy_path_or_list)
        else:
            box, coord, types, types_chem = read_npy(npy_path_or_list)
        (len_alist, len_atoms) = coord.shape[0:2]

        ## Check the system's validity
        max_cutoff = get_max_cutoff(box)
        self.log('\n >>> Recommendation) "min(max(r) in a fgpt) in the training set" < min(interplane distances)/2. <<<', no_space=True)
        self.log('*** min(interplane distance) / 2. == ({}) (calculated by all the boxes.)'.format(max_cutoff), no_space=True)
        # Check system size.
        supercell_tile = 1
        if len_atoms < self.num_cutoff and self.num_cutoff < 8*len_atoms and allow_supercell:
            message = (
                '\nWarning) The size of systems is too small when compared with the num_cutoff.'
                '\n*** num_cutoff == {}'
                '\n*** len_atoms == {}'
                '\n*** I will make fgpts after the systems are supercelled (2x2x2, i.e. 8 times).'
                '\n*** Must consider the consequencies.'
                .format(self.num_cutoff, len_atoms)
                )
            self.log(message, no_space=True)
            supercell_tile = 8
        elif len_atoms < self.num_cutoff and not allow_supercell:
            message = (
                '\nError) The size of systems is too small when considered to the num_cutoff.'
                '\n*** num_cutoff == {}'
                '\n*** len_atoms == {}'
                '\n*** {num_cutoff < 8*len_atoms} must be satisfied.'
                '\n*** You can trick it by making training sets as their supercells.'
                .format(self.num_cutoff, len_atoms)
                )
            if not allow_supercell:
                message += '\n*** Or {allow_supercell=True} would solve the problem.'
            self.log(message, no_space=True)
            raise RuntimeError(message)

        self.log('Getting fgpts...', tic='gen_fgpts', no_space=True)
        #### Get Fgpts & its Derivs.
        fgpt       = []
        fgpt_deriv = []
        neigh_ind  = []
        rot_mat    = []
        max_radi   = []
        if supercell_tile != 1:
            box, coord = make_x2_supercell(box, coord)
        # from time import time
        # time_ref = time()
        for i in range(len_alist):
            # print('_{} {}'.format(i, time() - time_ref))
            # time_ref = time()
            box_i   = box[i]
            coord_i = coord[i]
            # Rotational variation for a cell
            if rotational_variation:
                axis1 = np.random.rand(3)-0.5
                axis2 = np.random.rand(3)-0.5
                #--> shape of (3, 3)
                R = get_new_axis(axis1, axis2)
                rot_mat.append(R)
                box_i = (box_i @ R.T)

            #### Get fingerprint for one image
            fgpt_i       = []
            fgpt_deriv_i = []
            neigh_ind_i  = []
            for origin_atom in range(len_atoms):
                #### Get fingerprint of one atom in an image
                # Get relative coordinates in xyz coord.
                # And let origin_atom be at the center of the cell.
                #--> shape of (len_atoms, 3)
                rel_coord = ((coord_i - coord_i[origin_atom] + \
                    np.array([0.5, 0.5, 0.5], dtype=self.dtype)) % 1.0 - np.array([0.5, 0.5, 0.5])) @ box_i
                #--> shape of (len_atoms, 1)
                dist_vec = np.expand_dims(
                    np.linalg.norm(rel_coord, axis=1),
                    axis=1,
                    )
                #--> shape of (len_atoms, 6) c.f. second axis: (x, y, z, r, type, ind)
                prop_concat = np.concatenate(
                    (
                        rel_coord,
                        dist_vec,
                        #--> shape of (len_atoms, 1)
                        np.expand_dims(
                            np.tile(types, [supercell_tile]),
                            axis=1,
                            ),
                        #--> shape of (len_atoms, 1)
                        np.expand_dims(
                            list(range(len_atoms))*supercell_tile,
                            axis=1,
                            ),
                        ),
                    axis=1,
                    )

                # Sort in distance order
                prop_concat = prop_concat[prop_concat[:,3].argsort()]

                # Throw away center atom
                #--> shape of (num_cutoff, 6)
                prop_concat = prop_concat[range(1, self.num_cutoff+1)]
                max_radi.append(prop_concat[-1,3])

                #### Gatherings (atoms)
                # Gather fgpts
                # Now descriptor has atomic number as first element
                fgpt_i.append(
                    #--> shape of (num_cutoff, len_dscrptr)
                    np.concatenate(
                        (
                            #--> shape of (num_cutoff, 1)
                            np.expand_dims(prop_concat[:,4], axis=1),
                            #--> shape of (num_cutoff, len_dscrptr -1)
                            self.get_multipole_fgpt(
                                prop_concat[:,0:3],
                                np.expand_dims(prop_concat[:,3], axis=1),
                                ),
                            ),
                        axis=1,
                        ),
                    )

                # Gather neighbor list
                neigh_ind_i.append(
                    #--> shape of (num_cutoff,)
                    prop_concat[:,5]
                    )
                
                # Gather fgpt_derivs
                # c.f. 3 on second axis is for x, y, z
                # Derivatives of atomic numbers are zero (zeros concatenated as first elements)
                fgpt_deriv_i.append(
                    #--> shape of (num_cutoff, 3, len_dscrptr)
                    np.concatenate(
                        (
                            #--> shape of (num_cutoff, 3, 1)
                            np.expand_dims(
                                np.zeros((self.num_cutoff, 3)),
                                axis=2,
                                ),
                            #--> shape of (num_cutoff, 3, len_dscrptr -1)
                            self.get_multipole_fgpt_deriv(
                                prop_concat[:,0:3],
                                np.expand_dims(prop_concat[:,3], axis=1),
                                ),
                            ),
                        axis=2,
                        ),
                    )

            #### Gathering (imgs)
            fgpt.append(fgpt_i)
            fgpt_deriv.append(fgpt_deriv_i)
            neigh_ind.append(neigh_ind_i)

        min_max_radi = np.amin(max_radi)
        self.log('got fgpts...', toc='gen_fgpts', no_space=True)
        self.log(' >>> Check carefully) min(max(r) in a fgpt) in the training set is... <<<', no_space=True)
        self.log('=================================================================', no_space=True)
        self.log('*** min(max(r)) == {}'.format(min_max_radi), no_space=True)
        self.log('*** Reminder) min(interplane distance) / 2. == ({})'.format(max_cutoff), no_space=True)
        if min_max_radi > max_cutoff:
            self.log('      !!WARNING!!', no_space=True)
            self.log('*** Are you sure it is sufficiently high?', no_space=True)
            self.log('*** If not, bring the training set of bigger systems.', no_space=True)
            self.log('*** We recommend { min(max(r)) < min(interplane distance) / 2. }', no_space=True)
        self.log('=================================================================\n', no_space=True)
        # Output shapes
        #   fgpt       --> (len_alist, len_atoms, num_cutoff, len_dscrptr)
        #   fgpt_deriv --> (len_alist, len_atoms, num_cutoff, 3, len_dscrptr)
        #   neigh_ind  --> (len_alist, len_atoms, num_cutoff)
        #   rot_mat    --> (len_alist, len_atoms, 3, 3)
        #   types      --> (len_atoms)
        #   types_chem --> (len_atoms)
        if rotational_variation:
            return np.array(fgpt, dtype=self.dtype), np.array(fgpt_deriv, dtype=self.dtype), np.array(neigh_ind, dtype=np.int32), \
                   np.array(rot_mat, dtype=self.dtype), types, types_chem
        else:
            return np.array(fgpt, dtype=self.dtype), np.array(fgpt_deriv, dtype=self.dtype), np.array(neigh_ind, dtype=np.int32), \
                   types, types_chem

