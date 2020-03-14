#!/usr/bin/env python

import numpy as np
import sys

def alist2numpy(alist):
    if hasattr(alist[0]._calc, 'results'):
        if 'energy' in alist[0]._calc.results.keys():
            e_info = True
        else:
            e_info = False

        if 'forces' in alist[0]._calc.results.keys():
            f_info = True
        else:
            f_info = False

        if 'stress' in alist[0]._calc.results.keys():
            s_info = True
        else:
            s_info = False
    else:
        (e_info, f_info, s_info) = (False, False, False)

    #
    box = []
    coord = []
    energy = []
    force = []
    stress = []
    if e_info and f_info and s_info:
        for atoms in alist:
            box.append(np.array(atoms.get_cell(), dtype='float64'))
            coord.append(np.array(atoms.get_scaled_positions(), dtype='float64'))
            energy.append(atoms.get_potential_energy())
            force.append(np.array(atoms.get_forces(), dtype='float64'))
            stress.append(np.array(atoms.get_stress(), dtype='float64'))
    elif e_info and f_info and not s_info:
        for atoms in alist:
            box.append(np.array(atoms.get_cell(), dtype='float64'))
            coord.append(np.array(atoms.get_scaled_positions(), dtype='float64'))
            energy.append(atoms.get_potential_energy())
            force.append(np.array(atoms.get_forces(), dtype='float64'))
    elif e_info and not f_info and s_info:
        for atoms in alist:
            box.append(np.array(atoms.get_cell(), dtype='float64'))
            coord.append(np.array(atoms.get_scaled_positions(), dtype='float64'))
            energy.append(atoms.get_potential_energy())
            stress.append(np.array(atoms.get_stress(), dtype='float64'))
    elif e_info and not f_info and not s_info:
        for atoms in alist:
            box.append(np.array(atoms.get_cell(), dtype='float64'))
            coord.append(np.array(atoms.get_scaled_positions(), dtype='float64'))
            energy.append(atoms.get_potential_energy())
    elif not e_info and f_info and s_info:
        for atoms in alist:
            box.append(np.array(atoms.get_cell(), dtype='float64'))
            coord.append(np.array(atoms.get_scaled_positions(), dtype='float64'))
            force.append(np.array(atoms.get_forces(), dtype='float64'))
            stress.append(np.array(atoms.get_stress(), dtype='float64'))
    elif not e_info and f_info and not s_info:
        for atoms in alist:
            box.append(np.array(atoms.get_cell(), dtype='float64'))
            coord.append(np.array(atoms.get_scaled_positions(), dtype='float64'))
            force.append(np.array(atoms.get_forces(), dtype='float64'))
    elif not e_info and not f_info and s_info:
        for atoms in alist:
            box.append(np.array(atoms.get_cell(), dtype='float64'))
            coord.append(np.array(atoms.get_scaled_positions(), dtype='float64'))
            stress.append(np.array(atoms.get_stress(), dtype='float64'))
    elif not e_info and not f_info and not s_info:
        for atoms in alist:
            box.append(np.array(atoms.get_cell(), dtype='float64'))
            coord.append(np.array(atoms.get_scaled_positions(), dtype='float64'))
    box = np.array(box, dtype='float64')
    coord = np.array(coord, dtype='float64')
    if e_info:
        energy = np.array(energy, dtype='float64')
    if f_info:
        force = np.array(force, dtype='float64')
    if s_info:
        stress = np.array(stress, dtype='float64')
    return box, coord, energy, force, stress

if __name__ == '__main__':
    print("\n\n")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$".center(120))
    print("            ___________________________           ".center(120))
    print(" __________|  C o d e  b y  Y.J. Choi  |_________ ".center(120))
    print("|______________ ssrokyz@gmail.com _______________|".center(120))
    print("")
    print("*******   This code will generate npy files from trajectory file   *******".center(120))
    print("useage ==> ./traj2npy.py 'file' 'training/validation set ratio' 'shuffle or not(o/x)'".center(120))
    print("EXAMPLE) ./traj2npy.py GST_ran.traj 4 o".center(120))
    print("")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$".center(120))
    print("")
    if len(sys.argv) is 4:
        print(("The Number of arguments(= %d) is correct." %(len(sys.argv)-1)).center(120))
        print("\n")
    else:
        print("*****ERROR***** The number of arguments is not correct *****ERROR*****".center(120))
        print("\n")
        sys.exit(1)

    traj_file = sys.argv[1]
    t2v_ratio = float(sys.argv[2])
    if sys.argv[3] == 'o':
        shuffle = True
    elif sys.argv[3] == 'x':
        shuffle = False
    else:
        raise ValueError('Shuffle argument you gave is somehow wrong. It should be o or x. Please check.')

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$".center(120))
    print('')
    print(('file name: '+traj_file).center(120))
    print(('training/validation set ratio: '+str(t2v_ratio)).center(120))
    print(('shuffle: '+str(shuffle)).center(120))
    print('')
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$".center(120))
    print('')

    from time import time
    time_i = time()

    import subprocess as sp
    sp.call(['rm -rf old_npys'], shell=True)
    sp.call(['mv npys old_npys'], shell=True)
    sp.call(['mkdir -p npys/training npys/validation'], shell=True)

    from ase.io import read
    alist = read(traj_file, index=':', format='traj')
    image_num = len(alist)
    valid_num = int(image_num/(t2v_ratio+1))
    train_num = image_num - valid_num

    log_f = open('npys/log.txt', 'w')
    log_f.write('Made by file, ../'+traj_file+'\n')
    log_f.write('Number of total images: '+str(image_num)+'\n')
    log_f.write('Ratio of training/validation sets: ('+str(t2v_ratio)+' : 1)\n')
    log_f.write('Shuffle: '+str(shuffle)+'\n')
    log_f.write('Number of training sets:   '+str(train_num)+'\n')
    log_f.write('Number of validation sets: '+str(valid_num)+'\n')
    if shuffle:
        from random import shuffle as sffl
        sffl(alist)
    else:
        log_f.write('################################## Caution ####################################\n')
        log_f.write("     You didn't have order shuffled. Please be aware of what you're doing!     \n")
        log_f.write('################################## Caution ####################################\n\n')

    box, coord, energy, force, stress = alist2numpy(alist)

    np.save('npys/training/box.npy', box[:train_num])
    np.save('npys/validation/box.npy', box[train_num:])
    np.save('npys/training/coord.npy', coord[:train_num])
    np.save('npys/validation/coord.npy', coord[train_num:])
    if len(energy) != 0:
        np.save('npys/training/energy.npy', energy[:train_num])
        np.save('npys/validation/energy.npy', energy[train_num:])
    else:
        log_f.write(' *** energy information not exist *** \n')
        print(' *** energy information not exist *** '.center(120))
    if len(force) != 0:
        np.save('npys/training/force.npy', force[:train_num])
        np.save('npys/validation/force.npy', force[train_num:])
    else:
        log_f.write(' *** forces information not exist *** \n')
        print(' *** forces information not exist *** '.center(120))
    if len(stress) != 0:
        np.save('npys/training/stress.npy', stress[:train_num])
        np.save('npys/validation/stress.npy', stress[train_num:])
    else:
        log_f.write(' *** stress information not exist *** \n')
        print(' *** stress information not exist *** '.center(120))
        print('')
    log_f.close()

    type_txt = open('npys/type.txt', 'w')
    from ss_util import list2numlist as l2nl
    symbols = alist[0].get_chemical_symbols()
    if alist[-1].get_chemical_symbols() != symbols:
        raise ValueError("Chemical symbols seem to be not consistent btw images. Please check")
    symbols_num = l2nl(list(symbols))
    for nums in symbols_num:
        type_txt.write(str(nums)+" ")
    type_txt.write('\n')
    for symbol in symbols:
        type_txt.write(str(symbol)+" ")
    type_txt.close()
    sp.call(['cp npys/type.txt npys/training/'], shell=True)
    sp.call(['cp npys/type.txt npys/validation/'], shell=True)

    from ase.io.trajectory import Trajectory as Traj   
    train_traj = Traj('npys/training/training_set.traj', 'w')
    valid_traj = Traj('npys/validation/validation_set.traj', 'w')
    for i in range(train_num):
        train_traj.write(alist[i])
    for i in range(train_num,train_num+valid_num):
        valid_traj.write(alist[i])

    time_f = time()
    time_d = time_f - time_i
    print(('Total time used: '+str(time_d)+' sec ').center(120))
    print('\n\n')
    log_f = open('npys/log.txt', 'a')
    log_f.write('Total time used: '+str(time_d)+' sec ')
    log_f.close()
