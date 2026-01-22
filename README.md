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

The generated data will be saved in the `Data_npz` folder.

### 3. Go to the PET folder to train or test the model
The main script is `PET_main.py`.

3-1. Run from scratch (training to testing)  
Set the variables `new_experiment`, `pre_train`, and `fine_tune` to True.

3-2. Test a pre-trained model only  
Use the pre-trained model in the `Trained_model` folder.
Set `new_experiment`, `pre_train`, and `fine_tune` to `False`, and keep `test` and `dynamic_analysis` set to `True`.

The test results are saved in the `plt_results` folder, and the dynamic analysis results are saved in the `Dynamic_analysis` folder.

##  GPU and Hardware Requirements

The PET model was developed and trained using the following workstation:

- CPU: Intel Xeon Gold 6226R @ 2.90 GHz  
- RAM: 251 GB  
- GPU: NVIDIA GeForce RTX 3090 (24 GB VRAM)

For inference and testing (e.g., running pretrained models or short sequences), high-end GPU resources are generally not required. However, training the model from scratch, is computationally demanding due to the quadratic memory complexity of the Transformer attention mechanism.

The hardware configuration listed above is recommended for full-scale training as reported in our study. If GPU memory is limited but training is still desired, users may reduce the maximum input sequence length by modifying the `max_length` parameter in the configuration file. This allows training on smaller GPUs at the cost of reduced temporal context.
