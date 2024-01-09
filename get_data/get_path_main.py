import os

if __name__=="__main__":

    rawfeature_dir = ["./rawfeature/ML_HEA_100_bridgenew2.1.npy",
            "./rawfeature/ML_HEA_110_bridgenew2.1.npy",
            "./rawfeature/ML_HEA_111_bridgenew2.1.npy",
            "./rawfeature/ML_HEA_211_edgenew2.1.npy",
            "./rawfeature/ML_HEA_211_hillnew2.1.npy",
            "./rawfeature/ML_HEA_211_summitnew2.1.npy",
            "./rawfeature/ML_HEA_211_valleynew2.1.npy",
            "./rawfeature/ML_HEA_532_higher_edgenew2.1.npy",
            "./rawfeature/ML_HEA_532_hill1new2.1.npy",
            "./rawfeature/ML_HEA_532_hill2new2.1.npy",
            "./rawfeature/ML_HEA_532_inner_higher_edgenew2.1.npy",
            "./rawfeature/ML_HEA_532_outer_higher_edgenew2.1.npy",
            "./rawfeature/ML_HEA_532_valley2new2.1.npy"]

    ph_dir = ["./ph/ML_HEA_100_bridge/",
            "./ph/ML_HEA_110_bridge/",
            "./ph/ML_HEA_111_bridge/",
            "./ph/ML_HEA_211_edge/",
            "./ph/ML_HEA_211_hill/",
            "./ph/ML_HEA_211_summit/",
            "./ph/ML_HEA_211_valley/",
            "./ph/ML_HEA_532_higher_edge/",
            "./ph/ML_HEA_532_hill1/",
            "./ph/ML_HEA_532_hill2/",
            "./ph/ML_HEA_532_inner_higher_edge/",
            "./ph/ML_HEA_532_outer_higher_edge/",
            "./ph/ML_HEA_532_valley2/"]

    rawfeature_simu_dir = ["./rawfeature_simu/ML_HEA_100_simulation.npy",
            "./rawfeature_simu/ML_HEA_110_simulation.npy",
            "./rawfeature_simu/ML_HEA_111_simulation.npy",
            "./rawfeature_simu/ML_HEA_211_edge_OH_simulation.npy",
            "./rawfeature_simu/ML_HEA_211_hill_OH_simulation.npy",
            "./rawfeature_simu/ML_HEA_211_summit_OH_simulation.npy",
            "./rawfeature_simu/ML_HEA_211_valley_OH_simulation.npy",
            "./rawfeature_simu/ML_HEA_532_higher_edge_OH_simulation.npy",
            "./rawfeature_simu/ML_HEA_532_hill1_OH_simulation.npy",
            "./rawfeature_simu/ML_HEA_532_hill2_OH_simulation.npy",
            "./rawfeature_simu/ML_HEA_532_inner_higher_edge_OH_simulation.npy",
            "./rawfeature_simu/ML_HEA_532_outer_higher_edge_OH_simulation.npy",
            "./rawfeature_simu/ML_HEA_532_valley2_OH_simulation.npy"]

    ph_simu_dir = ["./ph_simu/ML_HEA_100_bridgesimu/",
            "./ph_simu/ML_HEA_110_bridgesimu/",
            "./ph_simu/ML_HEA_111_bridgesimu/",
            "./ph_simu/ML_HEA_211_edgesimu/",
            "./ph_simu/ML_HEA_211_hillsimu/",
            "./ph_simu/ML_HEA_211_summitsimu/",
            "./ph_simu/ML_HEA_211_valleysimu/",
            "./ph_simu/ML_HEA_532_higher_edgesimu/",
            "./ph_simu/ML_HEA_532_hill1simu/",
            "./ph_simu/ML_HEA_532_hill2simu/",
            "./ph_simu/ML_HEA_532_inner_higher_edgesimu/",
            "./ph_simu/ML_HEA_532_outer_higher_edgesimu/",
            "./ph_simu/ML_HEA_532_valley2simu/"]

    for i in range(13):
        this_rawfeature_dir = rawfeature_dir[i]
        this_ph_dir = ph_dir[i]
        command = 'python pathhomology.py --input_data ' + this_rawfeature_dir + ' --save_name ' + this_ph_dir
        os.system(command)

    for i in range(13):
        this_rawfeature_dir = rawfeature_simu_dir[i]
        this_ph_dir = ph_simu_dir[i]
        command = 'python pathhomology.py --input_data ' + this_rawfeature_dir + ' --save_name ' + this_ph_dir
        os.system(command)