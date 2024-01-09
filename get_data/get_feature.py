import os
import numpy as np
from decimal import Decimal
import math

def dis(coor1, coor2):
    the_dis = math.sqrt(((coor1[0] - coor2[0]) ** 2) + ((coor1[1] - coor2[1]) ** 2) + ((coor1[2] - coor2[2]) ** 2))
    return the_dis

def min_k(num_list, dis):
    topn_list = []
    for num in range(len(num_list)):
        if num_list[num] <= dis:
            topn_list.append(num)
    return topn_list

def bridge_sort(bridge1_electronegativity, bridge2_electronegativity):
    feature1 = []
    if bridge1_electronegativity >= bridge2_electronegativity:
        feature1.append(bridge1_electronegativity)
        feature1.append(bridge2_electronegativity)
    if bridge1_electronegativity < bridge2_electronegativity:
        feature1.append(bridge2_electronegativity)
        feature1.append(bridge1_electronegativity)
    return feature1

def get_npy(dir_oh, second_name, dir_digraph_oh, munb):
    label = r1 + "_" + second_name[:-3]  # label1
    dir_s = dir + r1 + "/surf/surf/OSZICAR"

    fd_oh = open(dir_oh, "r")
    row_oh = fd_oh.readlines()
    row_oh = row_oh[-1:]
    row_oh = '-' + row_oh[0].split('.')[2].split('E')[0]
    e_oh = float(row_oh[:4] + '.' + row_oh[4:])

    fd_s = open(dir_s, "r")
    row_s = fd_s.readlines()
    row_s = row_s[-1:]
    row_s = '-' + row_s[0].split('.')[2].split('E')[0]
    e_s = float(row_s[:4] + '.' + row_s[4:])

    de_oh = Decimal(str(e_oh)) - Decimal(str(e_s)) + Decimal(str(10.01187)) # label2

    fd_digraph_oh = open(dir_digraph_oh, "r")
    row_digraph_oh = fd_digraph_oh.readlines()

    arix_x = row_digraph_oh[2][:-1].split(' ')
    row_x = []
    for item in arix_x:
        if item != '':
            row_x.append(float(item))

    arix_y = row_digraph_oh[3][:-1].split(' ')
    row_y = []
    for item in arix_y:
        if item != '':
            row_y.append(float(item))

    arix_z = row_digraph_oh[4][:-1].split(' ')
    row_z = []
    for item in arix_z:
        if item != '':
            row_z.append(float(item))

    ele_list0 = row_digraph_oh[5][:-1].split(' ')
    ele_list = []
    for item in ele_list0:
        if item != '':
            ele_list.append(item)

    num_list0 = row_digraph_oh[6][:-1].split(' ')
    num_list = []
    for item in num_list0:
        if item != '':
            num_list.append(int(item))

    whole_ele_list = []  # label3
    for (ele, num) in zip(ele_list, num_list):
        for i in range(num):
            whole_ele_list.append(ele)

    coor_list0 = row_digraph_oh[9:numb]

    coor_list = []  # label4
    for coor in coor_list0:
        corr_temp1 = [float(coor.split('  ')[1]) * i for i in row_x]
        corr_temp2 = [float(coor.split('  ')[2]) * i for i in row_y]
        corr_temp3 = [float(coor.split('  ')[3]) * i for i in row_z]
        result0 = [i + j for i, j in zip(corr_temp1, corr_temp2)]
        result = [i + j for i, j in zip(result0, corr_temp3)]
        coor_list.append(result)

    # O
    ele_O = whole_ele_list[-2]
    coor_O = coor_list[-2]

    # H
    ele_H = whole_ele_list[-1]
    coor_H = coor_list[-1]

    # 扩胞
    num_hold = len(whole_ele_list) - 2
    whole_ele_list = whole_ele_list[: -2]
    coor_list = coor_list[:-2]

    # x
    for i in range(num_hold):
        coor_list.append(list(map(lambda x: x[0] + x[1], zip(coor_list[i], row_x))))
        whole_ele_list.append(whole_ele_list[i])
        coor_list.append(list(map(lambda x: x[0] - x[1], zip(coor_list[i], row_x))))
        whole_ele_list.append(whole_ele_list[i])
    # y
    for i in range(num_hold):
        coor_list.append(list(map(lambda x: x[0] + x[1], zip(coor_list[i], row_y))))
        whole_ele_list.append(whole_ele_list[i])
        coor_list.append(list(map(lambda x: x[0] - x[1], zip(coor_list[i], row_y))))
        whole_ele_list.append(whole_ele_list[i])

    # z
    for i in range(num_hold):
        coor_list.append(list(map(lambda x: x[0] + x[1], zip(coor_list[i], row_z))))
        whole_ele_list.append(whole_ele_list[i])
        coor_list.append(list(map(lambda x: x[0] - x[1], zip(coor_list[i], row_z))))
        whole_ele_list.append(whole_ele_list[i])

    ele_electronegativity = []  # label5
    ele_radii = []  # label6 Allioger Batsanov
    for ele in whole_ele_list:
        if ele == 'Ru': # 44
            ele_electronegativity.append(1.2)
            ele_radii.append(1.25 * 1.15)

        if ele == 'Rh': # 45
            ele_electronegativity.append(1.5)
            ele_radii.append(1.25 * 1.15)

        if ele == 'Ir': # 77
            ele_electronegativity.append(2.5)
            ele_radii.append(1.22 * 1.15)

        if ele == 'Pd': # 46
            ele_electronegativity.append(0.5)
            ele_radii.append(1.2 * 1.15)

        if ele == 'Pt': # 78
            ele_electronegativity.append(1.4)
            ele_radii.append(1.23 * 1.15)

    dis_bridge = []
    for i in range(len(whole_ele_list)):
        dis_bridge.append(dis(coor_list[i], coor_O))
    list_num = min_k(dis_bridge, 2)
    bridge1_num = list_num[0]
    bridge2_num = list_num[1]
    bridge_ele = []
    bridge1_ele = whole_ele_list[bridge1_num]
    bridge_ele.append(bridge1_ele)
    bridge2_ele = whole_ele_list[bridge2_num]
    bridge_ele.append(bridge2_ele)
    bridge1_electronegativity = ele_electronegativity[bridge1_num]
    bridge2_electronegativity = ele_electronegativity[bridge2_num]
    bridge1_coor = coor_list[bridge1_num]
    bridge2_coor = coor_list[bridge2_num]
    bridge1_radii = ele_radii[bridge1_num]
    bridge2_radii = ele_radii[bridge2_num]
    feature1 = bridge1_electronegativity + bridge2_electronegativity

    whole_ele_list_fin = []
    ele_electronegativity_fin = []
    ele_radii_fin = []
    coor_list_fin = []
    core_coor = [(bridge1_coor[0]+bridge2_coor[0])/2, (bridge1_coor[1]+bridge2_coor[1])/2, (bridge1_coor[2]+bridge2_coor[2])/2]

    dis_the_cloundpoint = []
    for i in range(len(whole_ele_list)):
        dis_the_cloundpoint.append(dis(coor_list[i], core_coor))
    list_num = min_k(dis_the_cloundpoint, (1.25 + 1.25 + 1.22 + 1.2 + 1.23) * 0.8)
    for j in list_num:
        whole_ele_list_fin.append(whole_ele_list[j])
        ele_electronegativity_fin.append(ele_electronegativity[j])
        ele_radii_fin.append(ele_radii[j])
        coor_list_fin.append(coor_list[j])

    dighy = []
    for i in range(len(whole_ele_list_fin)):
        for j in range(i, len(whole_ele_list_fin)):
            if i != j:
                if dis(coor_list_fin[i], coor_list_fin[j]) <= (1 + 0.2) * (ele_radii_fin[i] + ele_radii_fin[j]):
                    if ele_electronegativity_fin[i] <= ele_electronegativity_fin[j]:
                        dighy.append([i + 1, j + 1])
                    if ele_electronegativity_fin[i] >= ele_electronegativity_fin[j]:
                        dighy.append([j + 1, i + 1])

    dis_lst_fin = []
    for i in range(len(whole_ele_list_fin)):
        dis_lst_fin.append(dis(coor_list_fin[i], coor_O))
    _, ele_electronegativity_fin = zip(*sorted(zip(dis_lst_fin, ele_electronegativity_fin)))
    ele_feature_fin = list(ele_electronegativity_fin)

    dis_tmp = []
    for i in range(len(whole_ele_list_fin)):
        dis_tmp.append(dis(coor_list_fin[i], coor_O))
    list_num = min_k(dis_tmp, 2)
    bridge1_num = list_num[0]
    bridge2_num = list_num[1]

    dighy.append([bridge1_num + 1, len(whole_ele_list_fin) + 1])
    dighy.append([bridge2_num + 1, len(whole_ele_list_fin) + 1])
    dighy.append([len(whole_ele_list_fin) + 2, len(whole_ele_list_fin) + 1])

    whole_ele_list_fin.append(ele_O)
    whole_ele_list_fin.append(ele_H)
    coor_list_fin.append(coor_O)
    coor_list_fin.append(coor_H)

    the_HEA = []
    coor_list_fin = np.array(coor_list_fin)
    dighy = np.array(dighy)
    the_HEA.append(label)
    the_HEA.append(coor_list_fin)
    the_HEA.append(dighy)


    the_HEA.append(bridge_ele)
    the_HEA.append(feature1)
    the_HEA.append(list_num)
    the_HEA.append(whole_ele_list_fin)
    the_HEA.append(ele_feature_fin)
    the_HEA.append(float(de_oh))

    all_data.append(the_HEA)

    return len(whole_ele_list_fin)

data = [("./ML_HEA_100/", 75, 'bridge_OH'),
        ("./ML_HEA_111/", 75, 'bridge_OH'),
        ("./ML_HEA_110/", 95, 'bridge_OH'),
        ("./ML_HEA_211/", 107, 'edge_OH'),
        ("./ML_HEA_211/", 107, 'hill_OH'),
        ("./ML_HEA_211/", 107, 'valley_OH'),
        ("./ML_HEA_211/", 107, 'summit_OH'),
        ("./ML_HEA_532/", 104, 'higher_edge_OH'),
        ("./ML_HEA_532/", 104, 'inner_higher_edge_OH'),
        ("./ML_HEA_532/", 104, 'hill1_OH'),
        ("./ML_HEA_532/", 104, 'valley2_OH'),
        ("./ML_HEA_532/", 104, 'hill2_OH'),
        ("./ML_HEA_532/", 104, 'outer_higher_edge_OH')]

for the_npy in data:
    dir = the_npy[0]
    numb = the_npy[1]
    dir_list = []
    for root, dirs, files in os.walk(dir, topdown=True):
        dir_list.append(dirs)
    dir_list = dir_list[0]

    name2 = the_npy[2]
    all_data = []
    the_len = []
    for r1 in dir_list:
        dir_oh = dir + r1 + "/" + name2 + "/" + name2 + "/" + "OSZICAR"
        if os.path.exists(dir_oh):
            dir_digraph_oh = dir + r1 + "/" + name2 + "/" + name2 + "/" + "CONTCAR"
            out = get_npy(dir_oh, name2, dir_digraph_oh, numb)
            the_len.append(out)

    np.save('./rawfeature/' + dir[2:-1] + '_' + name2[:-3] + 'new2.1.npy', all_data)