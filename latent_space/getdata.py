import numpy as np

def returndata():
    data_dir1 = './train_data_new/'
    data_dir2 = './train_simu_data_new/'
    class_name = ['ML_HEA_100_bridge', 'ML_HEA_110_bridge', 'ML_HEA_111_bridge', 'ML_HEA_211_edge', 'ML_HEA_211_hill', 'ML_HEA_211_summit', 'ML_HEA_211_valley', 'ML_HEA_532_higher_edge', 'ML_HEA_532_hill1', 'ML_HEA_532_hill2', 'ML_HEA_532_inner_higher_edge', 'ML_HEA_532_outer_higher_edge', 'ML_HEA_532_valley2']
    label = []
    xdata = []
    ydata = []
    for the_class_name in class_name:
        dir1 = data_dir1 + the_class_name + '_fornn.npy'
        dir2 = data_dir2 + the_class_name + '_simu_fornn.npy'
        dataset = np.load(dir1, allow_pickle=True)
        x_data1 = dataset[0]
        y_data1 = dataset[1]
        dataset = np.load(dir2, allow_pickle=True)
        x_data2 = dataset[0]
        y_data2 = dataset[1]
        x_data = list(x_data1) + list(x_data2)
        y_data = list(y_data1) + list(y_data2)
        for index in x_data:
            bridge_ele = '?'
            if index[3] == round(1.2 + 1.2, 2): # 2.4
                bridge_ele = 'Ru-Ru'
            if index[3] == round(1.2 + 1.5, 2): # 2.7
                bridge_ele = 'Ru-Rh'
            if index[3] == round(1.2 + 2.5, 2): # 3.7
                bridge_ele = 'Ru-Ir'
            if index[3] == round(1.2 + 0.5, 2): # 1.7
                bridge_ele = 'Ru-Pd'
            if index[3] == round(1.2 + 1.4, 2): # 2.6
                bridge_ele = 'Ru-Pt'
            if index[3] == round(2.5 + 0.5, 2): # 3
                bridge_ele = 'Ir-Pd'
            if index[3] == round(1.5 + 1.5, 2) and index[4] == 1.5: # 3
                bridge_ele = 'Rh-Rh'
            if index[3] == round(1.5 + 2.5, 2): # 4
                bridge_ele = 'Rh-Ir'
            if index[3] == round(1.5 + 0.5, 2): # 2
                bridge_ele = 'Rh-pd'
            if index[3] == round(1.5 + 1.4, 2): # 2.9
                bridge_ele = 'Rh-Pt'
            if index[3] == round(2.5 + 2.5, 2): # 5
                bridge_ele = 'Ir-Ir'
            if index[3] == round(2.5 + 1.4, 2): # 3.9
                bridge_ele = 'Ir-Pt'
            if index[3] == round(0.5 + 0.5, 2): # 1
                bridge_ele = 'Pd-Pd'
            if index[3] == round(0.5 + 1.4, 2): # 1.9
                bridge_ele = 'Pd-Pt'
            if index[3] == round(1.4 + 1.4, 2): # 2.8
                bridge_ele = 'Pt-Pt'
            if bridge_ele == '?':
                print('>_<')
            label.append([bridge_ele, the_class_name])
        xdata += x_data
        ydata += y_data
    return xdata, ydata, label