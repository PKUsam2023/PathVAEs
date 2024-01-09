# Persistent Path Homology-based Semi-supervised Prediction and Generation Framework
This repo contains demonstrations of an a persistent path homology-based semi-supervised prediction and generation framework, empowered by our PathVAEs. This framework aims to predict the adsorption energy and design potential for High-entropy alloy catalysts.

This repository is adapted from the codebase used to produce the results in the paper "Path topology-assisted Semi-supervised Framework for High-Entropy Alloy Catalysts Prediction and Generation."

## Requirements

The code in this repo has been tested with the following software versions:
- Python>=3.7.0
- torch>=1.13.1
- numpy>=1.21.5
- scikit-learn>=0.24.2
- matplotlib>=3.3.4

The installation can be done quickly with the following statement.
```
pip install -r requirements.txt
```

We recommend using the Anaconda Python distribution, which is available for Windows, MacOS, and Linux. Installation for all required packages (listed above) has been tested using the standard instructions from the providers of each package.

## Data

The data for path complexes constructed for different HEA catalysts calculated by DFT is located in the directory
```
./get_data/rawfeature/
```
The data for path complexes constructed for simulation-generated different HEA catalysts is located in the directory
```
./get_data/rawfeature_simu/
```
The persistent path homology data, calculated from path complexes of different HEA catalysts derived from DFT calculations, can be found in the directory
```
./get_data/ph/
```
The persistent path homology data, calculated from path complexes of simulation-generated different HEA catalysts, can be found in the directory
```
./get_data/ph_simu/
```
The data for ligand features and coordination features of different HEA catalysts calculated by DFT, is available in the directory
```
./get_data/teacher_data/
```
The data for ligand features and coordination features of simulation-generated different HEA catalysts, is available in the directory
```
./get_data/student_data/
```
The data for ligand features and coordination features of different HEA catalysts calculated by DFT for the VAE training, is available in the directory
```
./PathVAEs/train_data_new/
```
The data for ligand features and coordination features of simulation-generated different HEA catalysts for the VAE training, is available in the directory
```
./PathVAEs/train_simu_data_new/
```
To obtain full data, please contact 2101212695@stu.pku.edu.cn 


## Files

This repo should contain the following files:
- 1 ./get_data/get_feature.py - The code employed in the retrieval of data pertains to the path complexes derived from DFT calculations for various HEA catalysts.
- 2 ./get_data/get_simulation.py - The code utilized for acquiring data corresponds to the path complexes generated through simulations for distinct HEA catalysts.
- 3 ./get_data/get_path_main.py - The code implemented is designed for the extraction of persistent path homology concerning diverse HEA catalysts. This encompasses catalysts for which calculations were performed using DFT, as well as those generated through simulations.
- 4 ./get_data/pathhomology.py - The method code for computing persistent path homology for path complexes.
- 5 ./get_data/get_feature_for_GBRT.py - The code employed to get the ligand features and coordination features of different HEA catalysts calculated by DFT.
- 6 ./get_data/get_feature_for_GBRT_test.py - The code employed to get the ligand features and coordination features of simulation-generated different HEA catalysts.
- 7 ./GBRT/teacher_data/ - The same to ./get_data/teacher_data/.
- 8 ./GBRT/student_data/ - The same to ./get_data/student_data/.
- 9 ./GBRT/getdata.py - The code employed for the data for the inputs to the GBRT model.
- 10 ./GBRT/GBRT.py - The main code for GBRT model.
- 11 ./GBRT/model.py - The code for GBRT model.
- 12 ./GBRT/model_test.py - The code employed for acquiring the labels of simulation-generated different HEA catalysts by trained GBRT.
- 13 ./nn/train_data/ - The same to ./get_data/teacher_data/.
- 14 ./nn/train_simu_data/ - The data of simulation-generated different HEA catalysts with labels acquired by trained GBRT.
- 15 ./nn/getdata.py - Method code for inputting data to the neural network.
- 16 ./nn/the_nn.py - The code for a neural network without semi-supervised learning.
- 17 ./PathVAEs/train_data/ - The same to ./nn/train_data/.
- 18 ./PathVAEs/train_simu_data/ - The same to ./nn/train_simu_data/.
- 19 ./PathVAEs/train_data_new/ - The data organized from ./PathVAEs/train_data/ for PathVAEs.
- 20 ./PathVAEs/train_simu_data_new/ - The data organized from ./PathVAEs/train_simu_data/ for PathVAEs.
- 21 ./PathVAEs/data_do.py - The code employed to get ./PathVAEs/train_data_new/ and ./PathVAEs/train_simu_data_new/ from ./PathVAEs/train_data/ and ./PathVAEs/train_simu_data/.
- 22 ./PathVAEs/getdata.py - Method code for inputting data to PathVAEs.
- 23 ./PathVAEs/the_nn.py - The code for PathVAEs with semi-supervised learning.
- 24 ./latent_space/image/ - The images of the latent spaces of PathVAEs.
- 25 ./latent_space/train_data_new/ - The same to ./PathVAEs/train_data_new/.
- 26 ./latent_space/train_simu_data_new/ - The same to ./PathVAEs/train_simu_data_new/.
- 27 ./latent_space/getdata.py - Method code for inputting data to PathVAEs.
- 28 ./latent_space/get_space.py - The code employed to get the latent spaces of PathVAEs and the high potential HEA catalyst structures.
- 29 ./latent_space/get_result_HEA/result_potential_HEA.npy - The acquired high potential HEA catalyst structures.
- 30 ./latent_space/get_result_HEA/ML_HEA_532_getstruction.npy - Standard 532 outer higher edge structure data.
- 31 ./latent_space/get_result_HEA/get_struction.py - Code to get the complete files for the high potential HEA catalyst structures.
- 32 ./latent_space/get_result_HEA/get_potential_HEA/ - The folder where the high potential HEA catalyst structures are located, if you need VESTA visualization please change the file name to CONTCAR.

## Model
The trained model files of PathVAEs are at 

If you find any bugs or have questions, please contact 2101212695@stu.pku.edu.cn 