import numpy as np
from sklearn.externals import joblib

dir_pre = './student_data/'
class_name = ['ML_HEA_100_bridge', 'ML_HEA_110_bridge', 'ML_HEA_111_bridge', 'ML_HEA_211_edge', 'ML_HEA_211_hill',
                  'ML_HEA_211_summit', 'ML_HEA_211_valley', 'ML_HEA_532_higher_edge', 'ML_HEA_532_hill1',
                  'ML_HEA_532_hill2', 'ML_HEA_532_inner_higher_edge', 'ML_HEA_532_outer_higher_edge',
                  'ML_HEA_532_valley2']
for the_class_name in class_name:
    dir = dir_pre + the_class_name + '_fortest.npy'
    dataset = np.load(dir, allow_pickle=True)
    feature_list = dataset

    X_test = np.array(feature_list, dtype='float32')
    model = joblib.load('./model1.pkl')
    pred_test = model.predict(X_test)

    data0 = []
    feature_list0 = []
    for item in feature_list:
        feature_list0.append(list(item))
    data0.append(list(feature_list))
    data0.append(list(pred_test))
    np.save('./' + the_class_name + '_simu_fornn.npy', data0)