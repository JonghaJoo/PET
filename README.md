## Usage

### 1. Install PyTorch and the necessary dependencies

```bash
pip install -r requirements.txt
```
### 2. Run file generating numerical hysteresis data in Data_generation folder
1. Bilinear
2. BoucWen
3. BWBN
4. IMK
5. RO

Run
```bash
Testing_main.py
```

The generated data will be saved in the Data_npz folder.

### 3. Go to the PET folder to train or test the model
The main script is PET_main.py.

3-1. Run from scratch (training to testing)  
Set the variables new_experiment, pre_train, and fine_tune to True.

3-2. Test a pre-trained model only  
Use the pre-trained model in the Trained_model folder.
Set new_experiment, pre_train, and fine_tune to False, and keep test and dynamic_analysis set to True.

The test results are saved in the plt_results folder, and the dynamic analysis results are saved in the Dynamic_analysis folder.
