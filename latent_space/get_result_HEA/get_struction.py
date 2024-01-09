import numpy as np
import random
import math

def dis(coor1, coor2):
    the_dis = math.sqrt(((coor1[0] - coor2[0]) ** 2) + ((coor1[1] - coor2[1]) ** 2) + ((coor1[2] - coor2[2]) ** 2))
    return the_dis

def sort_and_get_indices(input_list):
    indices = sorted(range(len(input_list)), key=lambda k: input_list[k])
    return indices

def rad_list():
    ele = random.randint(0, 4)
    if ele == 0:
        ele = 'Ru'
    if ele == 1:
        ele = 'Rh'
    if ele == 2:
        ele = 'Ir'
    if ele == 3:
        ele = 'Pd'
    if ele == 4:
        ele = 'Pt'
    return ele

def get_struction(ele_list, iii):
    dir_struction = './ML_HEA_532_getstruction.npy'
    loadstruction = np.load(dir_struction, allow_pickle=True)
    row_x = loadstruction[0]
    row_y = loadstruction[1]
    row_z = loadstruction[2]
    the_struct = loadstruction[3]
    element = ele_list
    struct_ele_lst = ['X' for i in range(len(the_struct))]

    ele_coor_H = the_struct[-1]
    ele_coor_O = the_struct[-2]
    dis_list = []

    for elecoor in the_struct[:-2]:
        dis_list.append(dis(elecoor, ele_coor_H))

    indices = sort_and_get_indices(dis_list)

    for i in range(len(element) - 2):
        for j in indices:
            if i == j:
                struct_ele_lst[j] = element[i]

    struct_ele_lst[-1] = 'H'
    struct_ele_lst[-2] = 'O'

    for ele in range(len(struct_ele_lst)):
        if struct_ele_lst[ele] == 'X':
            struct_ele_lst[ele] = rad_list()

    dict = {}
    for key in struct_ele_lst:
        dict[key] = dict.get(key, 0) + 1

    Ru_coor = []
    Rh_coor = []
    Ir_coor = []
    Pd_coor = []
    Pt_coor = []
    O_coor = []
    H_coor = []

    for ele in range(len(struct_ele_lst)):
        if struct_ele_lst[ele] == 'Ru':
            Ru_coor.append(the_struct[ele])
        if struct_ele_lst[ele] == 'Rh':
            Rh_coor.append(the_struct[ele])
        if struct_ele_lst[ele] == 'Ir':
            Ir_coor.append(the_struct[ele])
        if struct_ele_lst[ele] == 'Pd':
            Pd_coor.append(the_struct[ele])
        if struct_ele_lst[ele] == 'Pt':
            Pt_coor.append(the_struct[ele])
        if struct_ele_lst[ele] == 'O':
            O_coor.append(the_struct[ele])
        if struct_ele_lst[ele] == 'H':
            H_coor.append(the_struct[ele])

    f = open("./get_potential_HEA/potential_surf_HEA" + str(iii), "w")
    f.write("532 Ru-Ru" + '\n')
    f.write("1.0" + '\n')
    f.write("       " + str(row_x[0]) + "       " + str(row_x[1]) + "       " + str(row_x[2]) + '\n')
    f.write("       " + str(row_y[0]) + "       " + str(row_y[1]) + "       " + str(row_y[2]) + '\n')
    f.write("       " + str(row_z[0]) + "       " + str(row_z[1]) + "       " + str(row_z[2]) + '\n')
    f.write('   ' + 'Ru' + '   ' + 'Rh' + '   ' + 'Ir' + '   ' + 'Pd' + '   ' + 'Pt' + '   ' + '\n')
    f.write('   ' +  str(dict['Ru']) + '   ' +  str(dict['Rh'])   + '   ' +  str(dict['Ir'])   + '   ' +  str(dict['Pd']) + '   ' + str(dict['Pt']) + '   ' + '\n')
    f.write('Selective' + '\n')
    f.write('Cartesian' + '\n')
    for coor in Ru_coor:
        f.write('	' + str(coor[0]) + '	' + str(coor[1]) + '	' + str(coor[2]) + '\n')
    for coor in Rh_coor:
        f.write('	' + str(coor[0]) + '	' + str(coor[1]) + '	' + str(coor[2]) + '\n')
    for coor in Ir_coor:
        f.write('	' + str(coor[0]) + '	' + str(coor[1]) + '	' + str(coor[2]) + '\n')
    for coor in Pd_coor:
        f.write('	' + str(coor[0]) + '	' + str(coor[1]) + '	' + str(coor[2]) + '\n')
    for coor in Pt_coor:
        f.write('	' + str(coor[0]) + '	' + str(coor[1]) + '	' + str(coor[2]) + '\n')
    f.close()

    f = open("./get_potential_HEA/potential_oh_HEA" + str(iii), "w")
    f.write("532 Ru-Ru" + '\n')
    f.write("1.0" + '\n')
    f.write("       " + str(row_x[0]) + "       " + str(row_x[1]) + "       " + str(row_x[2]) + '\n')
    f.write("       " + str(row_y[0]) + "       " + str(row_y[1]) + "       " + str(row_y[2]) + '\n')
    f.write("       " + str(row_z[0]) + "       " + str(row_z[1]) + "       " + str(row_z[2]) + '\n')
    f.write('   ' + 'Ru' + '   ' + 'Rh' + '   ' + 'Ir' + '   ' + 'Pd' + '   ' + 'Pt' + '   ' + 'O' + '   ' + 'H' + '\n')
    f.write('   ' +  str(dict['Ru']) + '   ' +  str(dict['Rh'])   + '   ' +  str(dict['Ir'])   + '   ' +  str(dict['Pd']) + '   ' + str(dict['Pt']) + '   ' + str(dict['O']) + '   ' + str(dict['H']) + '\n')
    f.write('Selective' + '\n')
    f.write('Cartesian' + '\n')
    for coor in Ru_coor:
        f.write('	' + str(coor[0]) + '	' + str(coor[1]) + '	' + str(coor[2]) + '\n')
    for coor in Rh_coor:
        f.write('	' + str(coor[0]) + '	' + str(coor[1]) + '	' + str(coor[2]) + '\n')
    for coor in Ir_coor:
        f.write('	' + str(coor[0]) + '	' + str(coor[1]) + '	' + str(coor[2]) + '\n')
    for coor in Pd_coor:
        f.write('	' + str(coor[0]) + '	' + str(coor[1]) + '	' + str(coor[2]) + '\n')
    for coor in Pt_coor:
        f.write('	' + str(coor[0]) + '	' + str(coor[1]) + '	' + str(coor[2]) + '\n')
    for coor in O_coor:
        f.write('	' + str(coor[0]) + '	' + str(coor[1]) + '	' + str(coor[2]) + '\n')
    for coor in H_coor:
        f.write('	' + str(coor[0]) + '	' + str(coor[1]) + '	' + str(coor[2]) + '\n')
    f.close()

random.seed(6)
dir_potential_HEA = './result_potential_HEA.npy'

loadData1 = np.load(dir_potential_HEA, allow_pickle=True)

iii = 1
for ele_list in loadData1:
    get_struction(ele_list, iii)
    iii += 1