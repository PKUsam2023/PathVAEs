import numpy as np

def get_estimation(lst):
    the_lst = []
    for index in range(1, 41):
        if lst[index*2] == lst[index*2 - 1]:
            the_lst.append(lst[index*2])
        else:
            the_lst.append(lst[(index-1)*2])
    return the_lst

def get_graphy(this_ph, name):
    dir = this_ph + name + '.npy'
    loadData_graphy = np.load(dir, allow_pickle=True)
    a_list = []
    for i in loadData_graphy.item()['betti_num']:
        a_list.append(i.tolist())
    betti0 = [p[0] for p in a_list]
    betti1 = [p[1] for p in a_list]
    betti2 = [p[2] for p in a_list]
    betti0 = get_estimation(betti0)
    betti1 = get_estimation(betti1)
    betti2 = get_estimation(betti2)
    the_feature = [np.mean(betti0), np.mean(betti1), np.mean(betti2)]
    return the_feature

def get_data_test(dir_ph_test, dir_raw_test, the_class_name):
    this_ph = dir_ph_test + the_class_name + 'simu/'
    this_rawfeature = dir_raw_test + the_class_name + 'simu.npy'
    loadData1 = np.load(this_rawfeature, allow_pickle=True)
    name = [i[0] for i in loadData1]
    bridgr_electronegativity = [i[4] for i in loadData1]
    ele_list_electronegativity = [i[7] for i in loadData1]

    graphy = []
    fin_ele_list_electronegativity = []
    brig_electronegativity = []

    for i in range(len(name)):
        graphy.append(get_graphy(this_ph, name[i]))
        brig_electronegativity.append([bridgr_electronegativity[i]])
        fin_ele_list_electronegativity.append(ele_list_electronegativity[i])
    for i in range(len(name)):
        the_feature_zero = [0 for _ in range(37)]
        the_feature = [bridgr_electronegativity[i]] + graphy[i] + fin_ele_list_electronegativity[i]
        for i in range(len(the_feature)):
            the_feature_zero[i] = the_feature[i]
        all_feature_test.append(the_feature_zero)

dir_ph_pre = './ph/'
dir_raw_test = './rawfeature_simu/'
dir_ph_test = './ph_simu/'
class_name = ['ML_HEA_100_bridge', 'ML_HEA_110_bridge', 'ML_HEA_111_bridge', 'ML_HEA_211_edge', 'ML_HEA_211_hill', 'ML_HEA_211_summit', 'ML_HEA_211_valley', 'ML_HEA_532_higher_edge', 'ML_HEA_532_hill1', 'ML_HEA_532_hill2', 'ML_HEA_532_inner_higher_edge', 'ML_HEA_532_outer_higher_edge', 'ML_HEA_532_valley2']

for the_class_name in class_name:
    all_feature_test = []
    get_data_test(dir_ph_test, dir_raw_test, the_class_name)
    np.save('./student_data/' + the_class_name + '_fortest.npy', all_feature_test)