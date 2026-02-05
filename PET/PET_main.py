# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pretraining  import Transformer_pretraining, TransformerDataset, batch_padding, Trainer0
from finetuning import Transformer_finetuning, TransformerDataset, Trainer, Loss
from testing import Transformer_predict, Force_prediction
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import time
import math
import os
from backend import preprocess_data, downsampling, add_intervals_X, add_intervals_y
from config import device, embed_size, num_layers, forward_expansion, heads, MLP_size, max_length, root, input_dim, output_dim

#------------------------------------------
new_experiment = False
pre_train = False
fine_tune = False
#------------------------------------------
test = True
dynamic_preprocess = False
dynamic_analysis = True
hys_types = ['Bilinear', 'BoucWen', 'RO', 'BWBN', 'IMK'] 
for i in range(len(hys_types)):
    hys_type = hys_types[i]
    if new_experiment:
        save_dir = os.path.join(root,f'Newly_trained_model/{hys_type}')
    else:
        save_dir = os.path.join(root,f'Trained_model/{hys_type}')
    make_dir = os.makedirs(save_dir, exist_ok=True) 
    save_dir_dynamic = os.path.join(root,f'Dynamic_analysis/{hys_type}')
    make_dir = os.makedirs(save_dir_dynamic, exist_ok=True)
    save_fig_result = os.path.join(root,f'plt_results/{hys_type}')
    make_dir = os.makedirs(save_fig_result, exist_ok=True)
    save_param_path = os.path.join(save_dir, f'checkpoint_step_1_{hys_type}.pth')
    save_param_path2 = os.path.join(save_dir, f'checkpoint_step_2_{hys_type}.pth')
    save_loss_path = os.path.join(save_dir, f'{hys_type}_loss_checkpoint.pth')
    test_param = save_param_path2
    dynamic_analysis_param = save_param_path2
    Data = np.load(os.path.join(root,f'Data_npz/{hys_type}_PE_PI_DA2_Processed_data.npz'))
    X_test, X_val, y_test, y_val = Data['X_train'], Data['X_val'], Data['y_train'], Data['y_val']
    X_train, y_train = Data['X_test'], Data['y_test']

    X_train_tensor, y_train_tensor = preprocess_data(X_train,y_train, device)
    X_val_tensor, y_val_tensor = preprocess_data(X_val,y_val, device)
    X_test_tensor, y_test_tensor = preprocess_data(X_test,y_test, device)
    X_train_tensor_downsampled, y_train_tensor_downsampled = downsampling(X_train_tensor, y_train_tensor, max_length)
    X_val_tensor_downsampled, y_val_tensor_downsampled = downsampling(X_val_tensor, y_val_tensor, max_length,1,1)                      
    max_y=max([torch.max(y) for y in y_train_tensor_downsampled])+0.05
    min_y=min([torch.min(y) for y in y_train_tensor_downsampled])-0.05
    
    X_train_tensor_downsampled , y_train_tensor_downsampled = add_intervals_X(X_train_tensor_downsampled, device), add_intervals_y(y_train_tensor_downsampled, device)
    X_val_tensor_downsampled, y_val_tensor_downsampled  = add_intervals_X(X_val_tensor_downsampled, device), add_intervals_y(y_val_tensor_downsampled, device)
    X_test_tensor_downsampled , y_test_tensor_downsampled = add_intervals_X(X_test_tensor, device) , add_intervals_y(y_test_tensor, device)

    data_length_train = torch.tensor([s.shape[0] for s in X_train_tensor_downsampled], device=device)
    data_length_val = torch.tensor([s.shape[0] for s in X_val_tensor_downsampled], device=device)
    data_length_test = torch.tensor([s.shape[0] for s in X_test_tensor_downsampled], device=device)
    print(f'\n__Data Information for {hys_type}__')
    print('total number of training data: ', len(data_length_train))
    print('total number of validation data: ',len(data_length_val))
    print('total number of test data: ',len(data_length_test))
    
    train_dataset = TransformerDataset(X_train_tensor_downsampled, y_train_tensor_downsampled, data_length_train)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=batch_padding) 
    val_dataset = TransformerDataset(X_val_tensor_downsampled, y_val_tensor_downsampled, data_length_val)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=batch_padding)
    test_dataset = TransformerDataset(X_test_tensor_downsampled, y_test_tensor_downsampled, data_length_test)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=batch_padding)
    
    model = Transformer_pretraining(
        input_dim, output_dim,
        embed_size, num_layers, 
        forward_expansion, heads,
        MLP_size, device, max_length
    ).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"학습 가능한 파라미터 개수: {trainable_params}")
    print(model)

    if pre_train:
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.9)
        criterion = nn.MSELoss()
        trainer0 = Trainer0(model, optimizer,criterion, device)
        num_epochs = 1000
        loss_trend , val_trend = [], []
        for epoch in range(num_epochs):   
            train_loss, train_predictions1, indices1, X_predictions1, true_y1= trainer0.train_epoch(train_dataloader)   
            val_loss, val_predictions1, val_indices1, X_val_predictions1, val_true_y1= trainer0.evaluate(val_dataloader)  
            scheduler.step()
            loss_trend.append(train_loss)
            val_trend.append(val_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}")
            if (epoch + 1) % 100 == 0 :
                param_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss_trend': loss_trend,
                'val_trend': val_trend
                 } 
                torch.save(param_state, save_param_path)

        mse=[]
        physics=[]
        physics_new=[]
        for i in range(len(indices1)):
            energy_predictions1 = model.energy(X_predictions1[i], train_predictions1[i])      
            normalization_mse, normalization_physics, normalization_physics_new = Loss.normalization_loss(X_predictions1[i], train_predictions1[i], true_y1[i], energy_predictions1, device, min_y, max_y)
            mse.append(normalization_mse)
            physics.append(normalization_physics)
            physics_new.append(normalization_physics_new)
        mean_mse = torch.mean(torch.tensor(mse)).float()
        mean_physics = torch.mean(torch.tensor(physics)).float()
        mean_physics_new = torch.mean(torch.tensor(physics_new)).float()
        normalization_physics = ((mean_physics)/mean_mse).float()
        normalization_physics_new = ((mean_physics_new)/mean_mse).float()
        normalization_mse = 1
        print(normalization_physics)
        
        param_state={
            'model_state_dict': model.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_trend': loss_trend,
            'val_trend': val_trend
        }  
        loss_state = {
            "mse_loss_init":normalization_mse,
            "phys_loss_init": normalization_physics,
            "phys_loss_new_init": normalization_physics_new
        }
        
        torch.save(param_state, save_param_path)
        torch.save(loss_state, save_loss_path)
        del train_predictions1, indices1, X_predictions1, true_y1, energy_predictions1
        torch.cuda.empty_cache() 
        
    if fine_tune:
        model = Transformer_finetuning(
            input_dim=input_dim, output_dim=output_dim,
            embed_size=embed_size, num_layers=num_layers, 
            forward_expansion=forward_expansion, heads=heads,
            MLP_size=MLP_size, device=device, max_length=max_length
        ).to(device)
        checkpoint = torch.load(save_param_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        alpha = 0.05
        beta = 0.1
        checkpoint_loss = torch.load(save_loss_path)
        normalization_mse = checkpoint_loss["mse_loss_init"]
        normalization_physics = checkpoint_loss["phys_loss_init"]
        normalization_physics_new = checkpoint_loss["phys_loss_new_init"]
        criterion = Loss(min_y, max_y)
        optimizer = optim.AdamW(model.parameters(), lr=0.0005)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.9)
        val_trend2, loss_trend2 = [], []
        trainer = Trainer(model, optimizer, device, min_y, max_y, alpha, beta, normalization_mse ,normalization_physics ,normalization_physics_new)
        num_epochs = 1000
    
        for epoch in range(num_epochs):   
            train_loss, train_predictions1, indices1, X_predictions1 = trainer.train_epoch(train_dataloader, criterion)   
            val_loss, val_predictions1, val_indices1, X_val_predictions1 = trainer.evaluate(val_dataloader, criterion)  
            loss_trend2.append(train_loss), val_trend2.append(val_loss)
            scheduler.step()
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}")
            if (epoch + 1) % 100 == 0 :
                param_state2 = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss_trend2': loss_trend2,
                'val_trend2': val_trend2
                 } 
                torch.save(param_state2, save_param_path2)
         
                        
        param_state2 = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_trend2': loss_trend2,
        'val_trend2': val_trend2
         } 
        torch.save(param_state2, save_param_path2)
        torch.cuda.empty_cache() 
        del train_predictions1, indices1, X_predictions1, val_predictions1, val_indices1, X_val_predictions1

    if test:
        print(f'\n__Testing for {hys_type}__')
        save_path_R = os.path.join(save_dir, f'{hys_type}_average_R_square_test.npy')
        save_path_max = os.path.join(save_dir, f'{hys_type}_max_force_errors.npy')
        save_path_force = os.path.join(save_dir, f'{hys_type}_force_prediction.npz')
        model = Transformer_predict(
            input_dim, output_dim,
            embed_size, num_layers, 
            forward_expansion, heads,
            MLP_size, device, max_length
        ).to(device)
        checkpoint = torch.load(test_param, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer = Force_prediction(model, device)
        
        if os.path.exists(save_path_R):
            average_R_square_test = np.load(save_path_R, allow_pickle=True).tolist()
        else:
            average_R_square_test = []
            
        if os.path.exists(save_path_max):
            max_force_errors = np.load(save_path_max, allow_pickle=True).tolist()
        else:
            max_force_errors = []
            
        if os.path.exists(save_path_force):
            with np.load(save_path_force, allow_pickle=True) as d:
                force_prediction = list(d['force_prediction']) 
        else:
            force_prediction = []

        for N in range(len(data_length_test)):
            start_time = time.time()  
            Data = X_test_tensor_downsampled[N].to(device)
            nPts = len(Data)
            Data = Data.unsqueeze(0)
            fs = torch.zeros(1, nPts, 1).to(device)
            print(f"Processing test sample {N} (length = {nPts})_{hys_type}...") 
            if nPts <= max_length:
                u_input = Data
                f_pred = trainer.predict_epoch(u_input)
                fs = f_pred[:,:,:1]
            else:
                u_input = Data[:,:max_length,:]
                f_pred = trainer.predict_epoch(u_input)
                fs[:,:max_length,:] = f_pred[:,:,:1]   
                for i in range(max_length, nPts):
                    print(f"Step {i+1}/{nPts}", end="\r")
                    u_seq = Data[:,:i+1,:]
                    u_length = u_seq.shape[1]
                    if u_length <= max_length:
                        data=u_seq
                        
                    elif u_length  > max_length: 
                        step = math.ceil((u_length-max_length)/(max_length-1))+1
                        n = u_length - ((max_length-1)*(step-1)+1)
                        front = u_seq[:,0:(step*n)+1:step,:]
                        back = u_seq[:,(step*n)+(step-1):u_length:step-1,:]
                        data= torch.cat((front, back),dim=1)     
                    u_input = data
                    with torch.no_grad():
                        f_pred = trainer.predict_epoch(u_input)
                    fs[:, i, :] = f_pred[:,-1, :1] 
                      
            u_np = Data[0,:,0].cpu().numpy()
            fs_np = fs[0,:,0].cpu().numpy()
            force_prediction.append(fs_np)
            
            real_fs_np = y_test_tensor_downsampled[N][:, 0].cpu().numpy()
            max_errors = (np.max(real_fs_np)-np.max(fs_np))/(np.max(real_fs_np))
            min_errors = (np.abs(np.min(real_fs_np)-np.min(fs_np)))/np.abs(np.min(real_fs_np))
            errors = max(max_errors, min_errors)
            print("Maximum force errors",  errors)
            max_force_errors.append(errors)
    
        
            R_square = 1-np.sum((real_fs_np-fs_np)**2)/np.sum((real_fs_np-np.mean(real_fs_np))**2)
            print(f"R_squre for test sample {N}: {R_square:.4f}")
            average_R_square_test.append(R_square)
            
            fig = plt.figure(figsize=(8, 6))
            plt.plot(u_np, real_fs_np, linestyle = '--',label='Reference', color='black',linewidth = 2)
            plt.plot(u_np, fs_np, label='Predicted', color='red',alpha = 0.6,linewidth = 1.5)
            plt.xlabel(r'Normalized $u$', fontsize=20)
            plt.ylabel(r'Normalized $f$', fontsize=20)  
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=15)
            plt.grid()
            fig.savefig(os.path.join(root,f"plt_results/{hys_type}/{hys_type}_test{N}.svg"), format='svg')   
            np.save(save_path_R, average_R_square_test)
            np.save(save_path_max, max_force_errors)
            np.savez(save_path_force,force_prediction = np.array(force_prediction, dtype=object))

    if dynamic_analysis:
        print(f'\n__Dynamic Analysis for {hys_type}__')
        meta_Data = np.load(os.path.join(root,f'Dynamic_analysis/{hys_type}/{hys_type}_PE_PI_DA2_meta.npz'))
        delta_t = meta_Data['dt_list']
        length = meta_Data['nPts_list']
        if dynamic_preprocess == True:
            max_force = []
            max_disp = []
            for i in range(len(delta_t)):
                Dataa = np.load(os.paht.join(f'Dynamic_analysis/{hys_type}/{hys_type}_PE_PI_DA2_{i}.npz'))
                max_disp.append(np.max(np.abs(Dataa['disp'])))
                max_force.append(np.max(np.abs(Dataa['force'])))
            max_force = np.max(max_force)
            max_disp = np.max(max_disp)
            np.save(os.paht.join(f'Dynamic_analysis/{hys_type}/hysteresis_normalization_factor_f.npy', max_force))
            np.save(os.paht.join(f'Dynamic_analysis/{hys_type}/hysteresis_normalization_factor_u.npy', max_disp))
            
        norm_u = np.load(os.path.join(root,f'Dynamic_analysis/{hys_type}/hysteresis_normalization_factor_u.npy'))
        norm_f = np.load(os.path.join(root,f'Dynamic_analysis/{hys_type}/hysteresis_normalization_factor_f.npy'))
        normalize_gap=0.1
        
        def make_input(u_seq):
            u_normed = u_seq / (norm_u * (1 + normalize_gap))
            u_diff = torch.diff(u_normed, dim=1, prepend=u_normed[:, :1, :])
            u_diff2 = torch.diff(u_diff, dim=1, prepend=u_diff[:, :1, :])
            return torch.cat([u_normed, u_diff, u_diff2], dim=2)
        
        model = Transformer_predict(
            input_dim=3, output_dim=2,
            embed_size=20, num_layers=5,
            forward_expansion=2, heads=5,
            MLP_size=MLP_size, device=device, max_length=max_length
        ).to(device)
        checkpoint = torch.load(dynamic_analysis_param, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer = Force_prediction(model, device)
        errors = []
        for N in range(5):
            print('Sample_number:', N)
            Data = np.load(os.path.join(root,f'Dynamic_analysis/EQ_DATA/{N}.npy'))
            nPts = length[N]
            print('nPts:', nPts)
            dt = delta_t[N]
            print('dt:', dt)
        
            EQ = torch.tensor(Data, dtype=torch.float32, device=device)  
            nPts = len(Data)
            t = torch.linspace(0, (nPts - 1) * dt, nPts, device=device)
    
            m = 1.0
            k_linear = 6.283**2
            Ccr = 0.0
            c = Ccr * 0.02
            
            u = torch.zeros((1, nPts, 1), device=device)
            udot = torch.zeros_like(u)
            uddot = torch.zeros_like(u)
            fs = torch.zeros_like(u)
    
            uddot[:, 0, :] = (m * EQ[0] - c * udot[:, 0, :] - fs[:, 0, :]) / m
    
            u[:, 1, :] = u[:, 0, :] + dt * udot[:, 0, :] + 0.5 * dt**2 * uddot[:, 0, :]
            
            for i in range(2, nPts):
                print(f"Step {i}/{nPts - 1}", end="\r")
    
                u_seq = u[:, :i+1, :]
                
                u_length = u_seq.shape[1]
                if u_length <= max_length:
                    data=u_seq
                    
                elif u_length  > max_length: 
                    step = math.ceil((u_length-max_length)/(max_length-1))+1
                    n = u_length - ((max_length-1)*(step-1)+1)
                    front = u_seq[:,0:(step*n)+1:step,:]
                    back = u_seq[:,(step*n)+(step-1):u_length:step-1,:]
                    data= torch.cat((front, back),dim=1)
                         
                u_input = make_input(data)
                with torch.no_grad():
                    f_pred = trainer.predict_epoch(u_input)
                fs[:, i, :] = f_pred[:, -1, :1] * (norm_f * (1 + normalize_gap))
    
                udot[:, i, :] = (u[:, i, :] - u[:, i-1, :]) / dt
    
                if i < nPts -1:
                    u[:, i+1, :] = 2 * u[:, i, :] - u[:, i-1, :] + (dt ** 2 / m) * (
                        EQ[i] - c * udot[:, i, :] - fs[:, i, :])
        
            u_np = u.squeeze().cpu().numpy()
            u_np_norm = u_np / (norm_u * (1 + normalize_gap))
            f_np = fs.squeeze().cpu().numpy()
            f_np_norm = f_np / (norm_f * (1 + normalize_gap))
            
            real_data = np.load(os.path.join(root,f'Dynamic_analysis/{hys_type}/{hys_type}_PE_PI_DA2_{N}.npz'))
            real_disp = real_data['disp'][::2]/(norm_u*(1 + normalize_gap))
            real_force = real_data['force'][::2]/(norm_f*(1 + normalize_gap))
            
            fig = plt.figure(figsize=(8, 6))
            plt.plot(real_disp, real_force, linestyle = '--',label='Reference', color='black',linewidth = 2)
            plt.plot (u_np_norm,f_np_norm, label='Predicted', color='red',alpha = 0.6,linewidth = 1.5)
            plt.title(f'Test {N}')
            plt.xlabel(r'Normalized $u$', fontsize=20)
            plt.ylabel(r'Normalized $f$', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=15)
            plt.grid()
            print(f'\n\n__Result Analysis of sample number {N} for hysteresis model {hys_type}__')
            print('Opensees_max_force:{:.6f}'.format(max(abs(real_force))))
            open_max_force=max(abs(real_force))
            pred_max_force=max(abs(f_np_norm))
            force_error_ratio = abs((open_max_force-pred_max_force)/open_max_force)
            print('Predicted_max_force (PET):{:.6f}'.format(max(abs(f_np_norm))))
            print('Opensees_max_disp={:.6f}'.format(max(abs(real_disp))))
            open_max_displacement=max(abs(real_disp))
            pred_max_displacement=max(abs(u_np_norm))
            displacement_error_ratio = abs((open_max_displacement-pred_max_displacement)/open_max_displacement)
            print('Predicted_max_disp (PET)={:.6f}'.format(max(abs(u_np_norm))))
            print('Error of maximum force:{:.4f}%'.format(force_error_ratio*100))
            print('Error of maximum Disp:{:.4f}%'.format(displacement_error_ratio*100))
            R_square_force = 1 - np.sum((real_force - f_np_norm[:len(real_force)]) ** 2) / np.sum((real_force - np.mean(real_force)) ** 2)
            print('R_square_force:{:.4f}'.format(R_square_force))
            R_square_disp = 1 - np.sum((real_disp - u_np_norm[:len(real_disp)]) ** 2) / np.sum((real_disp - np.mean(real_disp)) ** 2)
            print('R_square_disp:{:.4f}'.format(R_square_disp))
            print('-------------------------------------------------')
    
            fig.savefig(os.path.join(save_dir_dynamic, f'PE_PI_DA2_predict_{N}_Paper_plot.svg'),format='svg')
            if N == 0:
                errors = np.array([[displacement_error_ratio,force_error_ratio,R_square_disp,R_square_force]])
                np.save(os.path.join(save_dir_dynamic, 'errors.npy'), errors)
            else:
                errors = np.load(os.path.join(save_dir_dynamic, 'errors.npy'))
                new_errors=np.array([[displacement_error_ratio,force_error_ratio,R_square_disp,R_square_force]])
                errors = np.vstack((errors,new_errors))
                np.save(os.path.join(save_dir_dynamic, 'errors.npy'), errors)
      
        
