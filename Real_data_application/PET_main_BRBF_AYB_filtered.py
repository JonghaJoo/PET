import torch
from torch.utils.data import DataLoader
from Realdata_finetuning import Transformer_finetuning, TransformerDataset, Trainer, batch_padding
import numpy as np
import matplotlib.pyplot as plt
import os 
from backend import preprocess_data, add_intervals_X, add_intervals_y
from config import device, embed_size, num_layers, forward_expansion, heads, MLP_size, root, input_dim, output_dim
data_path = os.path.join(root, 'Real_data_application','Realdata_test_BRBF_AYB_filtered.npz')
Data = np.load(data_path)
X_scale_factor = Data['X_scale_factor']
y_scale_factor = Data['y_scale_factor']
X_test = Data['X_test']
y_test = Data['y_test']
del Data 
X_test_tensor, y_test_tensor,test_len = preprocess_data(X_test,y_test, device)
max_length = 4500
X_test_tensor_downsampled , y_test_tensor_downsampled = add_intervals_X(X_test_tensor, device) , add_intervals_y(y_test_tensor, device)
data_length_test = torch.tensor([s.shape[0] for s in X_test_tensor_downsampled], device=device)
print('Total test samples:',len(data_length_test))
test_dataset = TransformerDataset(X_test_tensor_downsampled, y_test_tensor_downsampled, data_length_test)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=batch_padding)
save_dir = os.path.join(root,f'Trained_model/AYB')
make_dir = os.makedirs(save_dir, exist_ok=True) 
save_param_path = os.path.join(save_dir, 'checkpoint_AYB_final_filtered.pth')
save_loss_path = os.path.join(save_dir, 'loss_checkpoint_AYB_final_filtered.pth')
save_fig_result = os.path.join(root,f'plt_results/AYB')
os.makedirs(save_fig_result, exist_ok=True)  
#-------------------------------------------------------------------------------------------------------------------
model = Transformer_finetuning(
            input_dim=input_dim, output_dim=output_dim,
            embed_size=embed_size, num_layers=num_layers, #or 2
            forward_expansion=forward_expansion, heads=heads,
            MLP_size=MLP_size, device=device, max_length=max_length
        ).to(device)
alpha = 0.05
beta = 0.1
checkpoint_loss = torch.load(save_loss_path, map_location=device)
normalization_mse = checkpoint_loss["mse_loss_init"]
normalization_physics = checkpoint_loss["phys_loss_init"]
normalization_physics_new = checkpoint_loss["phys_loss_new_init"]
checkpoint = torch.load(save_param_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
trainer = Trainer(model,  device, alpha, beta, normalization_mse ,normalization_physics ,normalization_physics_new)   
test_loss, test_predictions1, test_indices1, X_test_predictions1 = trainer.evaluate(test_dataloader)
print(f"Test Loss: {test_loss:.8f}")
print(f"Plotting results...")
with torch.no_grad():
    y = y_test_tensor_downsampled[0][:, 0]
    y_pred = torch.as_tensor(test_predictions1[0][0, :, 0], device=y.device, dtype=y.dtype)
    ss_res = torch.sum((y - y_pred)**2)
    ss_tot = torch.sum((y - y.mean())**2)
    R_square = 1 - ss_res / ss_tot
    max_y = torch.max(torch.abs(y))
    max_y_pred = torch.max(torch.abs(y_pred))
    max_absolute_error = torch.abs(max_y - max_y_pred) / max_y
print(f"R_square for test sample 0: {R_square:.4f}, Max Absolute Error: {max_absolute_error:.4f}")
fig = plt.figure(figsize=(8, 6))
plt.plot(X_test_tensor_downsampled[0][:, 0].cpu().numpy()*X_scale_factor, y_test_tensor_downsampled[0][:, 0].cpu().numpy()*y_scale_factor, '-k' , label='Experimental')
plt.plot(X_test_predictions1[0][0, :, 0].detach().cpu().numpy()*X_scale_factor, test_predictions1[0][0, :, 0].detach().cpu().numpy()*y_scale_factor, '--r', label='PET',alpha = 0.8)
plt.xlabel('Displacement (mm)',fontsize=18)
plt.ylabel('Force (kN)',fontsize=18)
plt.grid()
plt.legend()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=11)
fig.savefig(os.path.join(save_fig_result,"Experimental_data_test_AYB.svg"), format='svg', bbox_inches='tight')