import numpy as np

def gat_data():
    data_dir = './train_data/'
    class_name = ['ML_HEA_100_bridge', 'ML_HEA_110_bridge', 'ML_HEA_111_bridge', 'ML_HEA_211_edge', 'ML_HEA_211_hill', 'ML_HEA_211_summit', 'ML_HEA_211_valley', 'ML_HEA_532_higher_edge', 'ML_HEA_532_hill1', 'ML_HEA_532_hill2', 'ML_HEA_532_inner_higher_edge', 'ML_HEA_532_outer_higher_edge', 'ML_HEA_532_valley2']
    for the_class_name in class_name:
        dir = data_dir + the_class_name + '_fornn.npy'
        dataset = np.load(dir, allow_pickle=True)
        x_data = []
        for index in dataset[0]:
            x_str = index[1:4]
            x_ele = []
            x_ele.append(round(index[0],2))
            x_ele = x_ele + index[4:]
            x = x_str + x_ele
            x_data.append(x)
        y_data = dataset[1]
        data = []
        data.append(x_data)
        data.append(y_data)
        np.save('./train_data_new/' + the_class_name + '_fornn.npy', data)

def gat_data_simu():
    data_dir = './train_simu_data/'
    class_name = ['ML_HEA_100_bridge', 'ML_HEA_110_bridge', 'ML_HEA_111_bridge', 'ML_HEA_211_edge', 'ML_HEA_211_hill', 'ML_HEA_211_summit', 'ML_HEA_211_valley', 'ML_HEA_532_higher_edge', 'ML_HEA_532_hill1', 'ML_HEA_532_hill2', 'ML_HEA_532_inner_higher_edge', 'ML_HEA_532_outer_higher_edge', 'ML_HEA_532_valley2']
    for the_class_name in class_name:
        dir = data_dir + the_class_name + '_simu_fornn.npy'
        dataset = np.load(dir, allow_pickle=True)
        x_data = []
        for index in dataset[0]:
            x_str = index[1:4]
            x_ele = []
            x_ele.append(round(index[0],2))
            x_ele = list(x_ele) + list(index[4:])
            x = list(x_str) + list(x_ele)
            x_data.append(x)
        y_data = dataset[1]
        data = []
        data.append(x_data)
        data.append(y_data)
        np.save('./train_simu_data_new/' + the_class_name + '_simu_fornn.npy', data)

if __name__ == "__main__":
    gat_data_simu()