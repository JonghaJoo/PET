import torch
import torch.nn as nn
import numpy as np
import os
import Experiment_230508 as Experiment
import pandas as pd
import ReadRecord


def linear_protocol(a, b, period, repetitions):
    if period % 2 != 0:
        raise ValueError('period must be even')
    peaks = a * period / 2 * np.arange(repetitions * 2 + 1) + b
    peaks = [peak * (-1)**i for i, peak in enumerate(peaks)]
    values = [np.linspace(peaks[i], peaks[i+1], period//2, endpoint=False) for i in range(len(peaks)-1)]
    return np.concatenate(values)

def add_diff(X):
    X_diff = torch.diff(X, axis=1, prepend=torch.zeros(X.size(0), 1, X.size(2)).to(X.device))
    X = torch.cat([X, X_diff], axis=2)
    return X


def augmentation(sample, rate):
    raise NotImplementedError
    # sample: [batch, time, features]

    shifted_right = torch.roll(sample, shifts=1, dims=1)
    shifted_left = torch.roll(sample, shifts=-1, dims=1)
    local_max_mask = (sample > shifted_right) & (sample > shifted_left)

    local_min_mask = (sample < shifted_right) & (sample < shifted_left)

    extrema_mask = local_max_mask | local_min_mask

    non_extrema_mask = ~extrema_mask

    indices = torch.randperm(sample.size(1))

    num_augmented = int(sample.size(1) * rate)

    non_extrema_mask = non_extrema_mask[:, indices[:num_augmented], :]


class Loss:
    def __init__(self, y_pred, y, energies, mask, device):
        self.mse_loss_init = (torch.mean((y_pred - y)**2 * mask)).to(device).detach()
        self.phys_loss_init = (self.drucker_loss(y_pred, energies, mask)).to(device).detach()
        print('MSE Loss / Phys Loss will be normalized by {}/{}'.format(self.mse_loss_init, self.phys_loss_init))
        pass

    def combined_loss(self, y_pred, y, energies, mask, alpha=0.2):
        self.mse_loss = torch.mean((y_pred - y)**2 * mask)
        self.phys_loss = self.drucker_loss(y_pred, energies, mask)   # Assuming y_pred and y are 2D tensors [batch_size x seq_length]
        self.total_loss = (1-alpha) * self.mse_loss / self.mse_loss_init + alpha * self.phys_loss / self.phys_loss_init
        print('Total Loss / MSE Loss / Phys Loss: {:.8f}/{:.8f}/{:.8f}'.format(self.total_loss, self.mse_loss, self.phys_loss), end='\r')
        return self.total_loss
    
    def drucker_loss(self, f, e, mask):
    
        num_samples, time_length = f.size()
        num_chops = 201
        chop_vector = torch.linspace(-1, 1, num_chops).to(f.device)

        # Expand the mask and tensors along the second axis
        mask_expanded = mask.unsqueeze(1).expand(-1, chop_vector.size(0), -1)
        f_expanded = f.unsqueeze(1).expand(-1, chop_vector.size(0), -1)
        e_expanded = e.unsqueeze(1).expand(-1, chop_vector.size(0), -1)

        chop_vector_expanded = chop_vector.unsqueeze(0).unsqueeze(-1).expand(num_samples, -1, time_length)
        deducted_f_expanded = f_expanded - chop_vector_expanded

        # Create deducted_f_sign tensor with size [num_samples, num_chops, time_length]
        deducted_f_sign = (deducted_f_expanded > 0).int()

        diff_sign = torch.diff(deducted_f_sign, dim=2, prepend=deducted_f_sign[:, :, 0].unsqueeze(2))
        change_bool = ((diff_sign != 0) * mask_expanded).bool()
        change_idx = torch.nonzero(change_bool)
        selected_e = e_expanded[change_idx[:, 0], change_idx[:, 1], change_idx[:, 2]]
        diff_e = selected_e[1:] - selected_e[:-1]
        diff_idx = change_idx[1:] - change_idx[:-1]
        diff_e = diff_e * (diff_idx[:, 0] == 0) * (diff_idx[:, 1] == 0)
        diff_e_neg = diff_e[diff_e < 0]
        return -torch.sum(diff_e_neg) / (num_samples * time_length)



class CustomLSTM2(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2, output_dim):
        super(CustomLSTM2, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        
        self.cell = CustomLSTMCell(input_dim, hidden_dim)
        self.cell2 = CustomLSTMCell(hidden_dim, hidden_dim2)
        self.fc = nn.Linear(hidden_dim2 + 1, output_dim, bias=False)
        

    def forward(self, x, states=None):
        batch_size, seq_length, _ = x.size()

        if states is None:
            h = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            h2 = torch.zeros(batch_size, self.hidden_dim2).to(x.device)
            c = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            c2 = torch.zeros(batch_size, self.hidden_dim2).to(x.device)
            prev_output = torch.zeros(batch_size, 1).to(x.device)
            h_energy = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            h_energy2 = torch.zeros(batch_size, self.hidden_dim2).to(x.device)
            energy = torch.zeros(batch_size, 1).to(x.device)
            previous_x = torch.zeros(batch_size, 1).to(x.device)  # Initialize previous_x with zeros
        else:
            h, h2, c, c2, prev_output, h_energy, h_energy2, energy, previous_x = states

        outputs = []
        energies = []
        for t in range(seq_length):
            prev_h = h
            prev_h2 = h2
            h, c = self.cell(x[:, t, :], h_energy, (h, c))
            h2, c2 = self.cell2(h, h_energy2, (h2, c2))
            output = self.fc(torch.cat([h2, x[:, t, 0].unsqueeze(1)], dim=1)) # Need to check if this is correct
            
            # Calculate and accumulate energy using the trapezoid rule
            current_x = x[:, t, 0].unsqueeze(1)
            delta_disp = current_x - previous_x
            h_energy = h_energy + (h + prev_h) / 2 * delta_disp
            h_energy2 = h_energy2 + (h2 + prev_h2) / 2 * delta_disp
            energy = energy + (output + prev_output) / 2 * delta_disp
            energies.append(energy)
            prev_output = output
            previous_x = current_x
            outputs.append(output)

        return torch.stack(outputs, dim=1), torch.stack(energies, dim=1), (h, h2, c, c2, prev_output, h_energy, h_energy2, energy, previous_x)


class CustomLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Linear transformation for energy
        self.energy_transform = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        
        # Input gate
        self.fc_i = nn.Linear(self.input_dim + self.hidden_dim * 2, self.hidden_dim)  # *2 for energy
        # Forget gate
        self.fc_f = nn.Linear(self.input_dim + self.hidden_dim * 2, self.hidden_dim)
        # Cell state
        self.fc_c = nn.Linear(self.input_dim + self.hidden_dim * 2, self.hidden_dim)
        # Output gate
        self.fc_o = nn.Linear(self.input_dim + self.hidden_dim * 2, self.hidden_dim)

        self.ln_h = nn.LayerNorm(self.hidden_dim)

    def forward(self, x, h_energy, states):
        h, c = states
        h_normalized = self.ln_h(h)
        # Transform the energy
        transformed_energy = torch.tanh(self.energy_transform(h_energy))
        
        h_combined = torch.cat([x, transformed_energy, h_normalized], 1)  # concatenate along the feature dimension

        i = torch.sigmoid((self.fc_i(h_combined)))
        f = torch.sigmoid((self.fc_f(h_combined)))
        g = torch.tanh((self.fc_c(h_combined)))
        o = torch.sigmoid((self.fc_o(h_combined)))
        
        c_next = f * c + i * g
        h_next = self.ln_h(o * torch.tanh(c_next) + h)    # Adding residual connection

        return h_next, c_next


class CustomLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.cell = CustomLSTMCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim + 1, output_dim, bias=False)
        

    def forward(self, x, states=None):
        batch_size, seq_length, _ = x.size()

        if states is None:
            h = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            c = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            prev_output = torch.zeros(batch_size, 1).to(x.device)
            h_energy = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            energy = torch.zeros(batch_size, 1).to(x.device)
            previous_x = torch.zeros(batch_size, 1).to(x.device)  # Initialize previous_x with zeros
        else:
            h, c, prev_output, h_energy, energy, previous_x = states

        outputs = []
        energies = []
        for t in range(seq_length):
            prev_h = h
            h, c = self.cell(x[:, t, :], h_energy, (h, c))
            output = self.fc(torch.cat([h, x[:, t, 0].unsqueeze(1)], dim=1)) # Need to check if this is correct
            
            # Calculate and accumulate energy using the trapezoid rule
            current_x = x[:, t, 0].unsqueeze(1)
            delta_disp = current_x - previous_x
            h_energy = h_energy + (h + prev_h) / 2 * delta_disp
            energy = energy + (output + prev_output) / 2 * delta_disp
            energies.append(energy)
            prev_output = output
            previous_x = current_x
            outputs.append(output)

        return torch.stack(outputs, dim=1), torch.stack(energies, dim=1), (h, c, prev_output, h_energy, energy, previous_x)
    



class BasicLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BasicLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # LSTM gates
        self.fc_all = nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim * 4)
        self.ln_h = nn.LayerNorm(self.hidden_dim)

    def forward(self, x, init_states):
        h, c = init_states
        combined = torch.cat((x, h), 1)  # concatenate x and h
        
        # LSTM gates
        combined_gates = self.fc_all(combined)
        i, f, g, o = torch.split(combined_gates, self.hidden_dim, dim=1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        
        # Update the cell state and hidden state
        c_next = f * c + i * g
        h_next = self.ln_h(o * torch.tanh(c_next))
        
        return h_next, c_next

class PyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PyLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.cell1 = BasicLSTMCell(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.cell2 = BasicLSTMCell(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.norm_input = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.norm_output = nn.Linear(self.output_dim, self.output_dim, bias=False)
        self.norm_factor_input = nn.Parameter(torch.ones(input_dim), requires_grad=False)
        self.norm_factor_output = nn.Parameter(torch.ones(output_dim), requires_grad=False)

    def forward(self, x, states=None):
        batch_size, seq_length, _ = x.size()
        
        # Initialize h and c for the first timestep
        if states is None:
            h1 = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            h2 = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            c1 = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            c2 = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        else:
            h1, c1, h2, c2 = states
        
        outputs = torch.zeros(batch_size, seq_length, 1).to(x.device)
        for t in range(seq_length):
            h1, c1 = self.cell1(x[:, t, :], (h1, c1))
            h1_ = self.act1(h1)
            h2, c2 = self.cell2(h1_, (h2, c2))
            h2_ = self.act2(h2)
            h2_ = (h2_ * 1 + h1_ * 0.5) / 1.5
            output = self.fc(h2_)

            outputs[:, t, :] = output
        
        return outputs, outputs, (h1, c1, h2, c2)

class BasicLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BasicLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.cell = BasicLSTMCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, states=None):
        """
        forward method for the LSTM.
        """
        batch_size, seq_length, _ = x.size()
        
        # Initialize h and c for the first timestep
        if states is None:
            h = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            c = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            prev_output = torch.zeros(batch_size, 1).to(x.device)
            energy = torch.zeros(batch_size, 1).to(x.device)
            previous_x = torch.zeros(batch_size, 1).to(x.device)  # Initialize previous_x with zeros
        else:
            h, c, prev_output, energy, previous_x = states
        
        outputs = []
        energies = []        
        for t in range(seq_length):
            h, c = self.cell(x[:, t, :], (h, c))
            output = self.fc(h)
            current_x = x[:, t, 0].unsqueeze(1)
            delta_disp = current_x - previous_x
            energy = energy + (output + prev_output) / 2 * delta_disp
            energies.append(energy)
            prev_output = output
            previous_x = current_x
            outputs.append(output)
        
        # Stack outputs along the sequence dimension
        outputs = torch.stack(outputs, dim=1)
        energies = torch.stack(energies, dim=1)
        
        return outputs, energies, (h, c, prev_output, energy, previous_x)
    

def result_data(processed_data_dir, 
                Title,
                mat_type,
                mat_props,
                model_dir,
                loss_dir,
                device,
                save_data_dir):
    raise NotImplementedError
    Data = np.load(os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz')))
    X_max, y_max, normalize_gap = Data['X_max'], Data['y_max'], Data['normalize_gap']
    # Test for maximum displacement of 1.1 larger than it was in training
    disps = [disp * X_max * (1.1) for disp in disps]

    outputs = [Experiment.static_1DOF(mat_type, mat_props, disps[i]) for i in range(len(disps))]
    X_tests, y_tests = [disp / (X_max * (1 + normalize_gap)) for disp in disps], [output['force'] / (y_max * (1 + normalize_gap)) for output in outputs]
    n_samples = len(disps)

    X_test, y_test, time_length = [], [], []

    num_inputs = 2  # Displacement and mask
    for i in range(n_samples):
        disp, force = X_tests[i], y_tests[i]
        X_test.append(np.concatenate((disp[:, np.newaxis], np.ones((len(disp), 1))), axis=1))
        y_test.append(force)
        time_length.append(len(disp))
    time_length = np.array(time_length)
    max_time_length = np.max(time_length)

    for i in range(n_samples):
        if len(X_test[i]) < max_time_length:
            X_test[i] = np.concatenate((X_test[i], np.zeros((max_time_length - len(X_test[i]), num_inputs))), axis=0)
            y_test[i] = np.concatenate((y_test[i], np.zeros((max_time_length - len(y_test[i])))), axis=0)
    X_test, y_test = np.array(X_test), np.array(y_test)


    loss_data = pd.read_csv(os.path.normpath(os.path.join(loss_dir, Title + '_loss.csv')))
    losses = loss_data['Loss'].values

    best_model_idx = np.argmin(losses)
    best_model_epoch = loss_data['Epoch'].values[best_model_idx]
    checkpoint = torch.load(os.path.normpath(os.path.join(model_dir, './' + Title +'_checkpoint_{}.pth'.format(int(best_model_epoch)))))

    nn_size = checkpoint['cell.fc_f.bias'].size()[0]
    model = CustomLSTM(2, nn_size, 1)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # Load test data
    Data = np.load(os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz')))
    X_test, mask_test = X_test[:, :, :1], X_test[:, :, 1]
    X_test =torch.from_numpy(X_test).float().to(device)
    X_test = add_diff(X_test)
    y_test = torch.from_numpy(y_test).float().to(device)
    mask_test = torch.from_numpy(mask_test).float().to(device)

    y_test_pred, energies_test, _ = model(X_test)
    y_test_pred = y_test_pred.cpu().detach().numpy()
    X_test = X_test.cpu().detach().numpy()
    y_test = y_test.cpu().detach().numpy()
    mask_test = mask_test.cpu().detach().numpy()

    # Save test results
    np.savez(os.path.join(save_data_dir, Title + '_Cyclic.npz'), X_test=X_test, y_test=y_test, mask_test=mask_test, y_test_pred=y_test_pred)

def read_EQ(EQ_data_file):
    with open(EQ_data_file, 'r') as f:
        EQ_list_ = [line.strip() for line in f]
    print(EQ_list_[:3])

    EQ_list = []
    for i, EQ in enumerate(EQ_list_):
        EQ_name = './'+EQ
        EQ_list.append(os.path.normpath(os.path.join('/home/jaehwan/Python Project/DLCM/Data', EQ_name)).replace("\\", "/"))
    print(EQ_list[:3])

    gm_list, t_list = [], []
    for EQ_name in EQ_list:
        file_name, file_extension = os.path.splitext(EQ_name)
        dt, nPts = ReadRecord.ReadRecord(EQ_name, file_name+'.dat')
        file_name, file_extension = os.path.splitext(EQ_name)
        gm = np.load(file_name+'.npy')
        t = np.arange(0, nPts*dt, dt)
        gm_list.append(gm)
        t_list.append(t)
    return gm_list, t_list
        

def test_prediction(loss_dir,
                    Title,
                    model_dir,
                    processed_data_path,
                    save_data_path,
                    device,
                    model = False):
    loss_data = pd.read_csv(os.path.normpath(os.path.join(loss_dir, Title + '_loss.csv')))
    losses = loss_data['Loss'].values

    best_model_idx = np.argmin(losses)
    best_model_epoch = loss_data['Epoch'].values[best_model_idx]
    checkpoint = torch.load(os.path.normpath(os.path.join(model_dir, './' + Title +'_checkpoint_{}.pth'.format(int(best_model_epoch)))))
    if model == False:
        nn_size = checkpoint['cell.fc_f.bias'].size()[0]
        model = CustomLSTM(2, nn_size, 1)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # Load test data
    Data = np.load(processed_data_path)
    X_test, y_test = Data['X_test'], Data['y_test']
    del Data
    X_test, mask_test = X_test[:, :, :1], X_test[:, :, 1]
    X_test =torch.from_numpy(X_test).float().to(device)
    X_test = add_diff(X_test)
    y_test = torch.from_numpy(y_test).float().to(device)
    mask_test = torch.from_numpy(mask_test).float().to(device)

    y_test_pred, energies_test, _ = model(X_test)
    y_test_pred = y_test_pred.cpu().detach().numpy()
    X_test = X_test.cpu().detach().numpy()
    y_test = y_test.cpu().detach().numpy()
    mask_test = mask_test.cpu().detach().numpy()
    energies_test = energies_test.cpu().detach().numpy()

    # Save test results
    np.savez(save_data_path, X_test=X_test, y_test=y_test, mask_test=mask_test, y_test_pred=y_test_pred, energies_test=energies_test)


def generate_cyclic(mat_type,
                    mat_props,
                    disps,
                    processed_data_path,
                    processed_cyclic_path):
    Data = np.load(processed_data_path)
    X_max, y_max, normalize_gap = Data['X_max'], Data['y_max'], Data['normalize_gap']
    # Test for maximum displacement of 1.1 larger than it was in training
    disps = [disp * X_max * (1.1) for disp in disps]

    outputs = [Experiment.static_1DOF(mat_type, mat_props, disps[i]) for i in range(len(disps))]
    X_tests, y_tests = [disp / (X_max * (1 + normalize_gap)) for disp in disps], [output['force'] / (y_max * (1 + normalize_gap)) for output in outputs]
    n_samples = len(disps)

    X_test, y_test, time_length = [], [], []

    num_inputs = 2  # Displacement and mask
    for i in range(n_samples):
        disp, force = X_tests[i], y_tests[i]
        X_test.append(np.concatenate((disp[:, np.newaxis], np.ones((len(disp), 1))), axis=1))
        y_test.append(force)
        time_length.append(len(disp))
    time_length = np.array(time_length)
    max_time_length = np.max(time_length)

    for i in range(n_samples):
        if len(X_test[i]) < max_time_length:
            X_test[i] = np.concatenate((X_test[i], np.zeros((max_time_length - len(X_test[i]), num_inputs))), axis=0)
            y_test[i] = np.concatenate((y_test[i], np.zeros((max_time_length - len(y_test[i])))), axis=0)
    X_test, y_test = np.array(X_test), np.array(y_test)
    np.savez(processed_cyclic_path, X_test=X_test, y_test=y_test)




def drucker_loss_numpy(f, e, mask):
    num_samples, time_length = f.shape
    num_chops = 201
    chop_vector = np.linspace(-1, 1, num_chops)

    # Expand the mask and tensors along the second axis
    mask_expanded = np.expand_dims(mask, axis=1) * np.ones((num_samples, num_chops, time_length))
    f_expanded = np.expand_dims(f, axis=1) * np.ones((num_samples, num_chops, time_length))
    e_expanded = np.expand_dims(e, axis=1) * np.ones((num_samples, num_chops, time_length))

    chop_vector_expanded = np.expand_dims(np.expand_dims(chop_vector, axis=0), axis=-1) * np.ones((num_samples, num_chops, time_length))
    deducted_f_expanded = f_expanded - chop_vector_expanded

    # Create deducted_f_sign tensor with size [num_samples, num_chops, time_length]
    deducted_f_sign = (deducted_f_expanded > 0).astype(int)

    diff_sign = np.diff(deducted_f_sign, axis=2, prepend=deducted_f_sign[:, :, [0]])
    change_bool = (diff_sign != 0) & mask_expanded.astype(bool)
    change_idx = np.argwhere(change_bool)

    selected_e = e_expanded[change_idx[:, 0], change_idx[:, 1], change_idx[:, 2]]
    diff_e = selected_e[1:] - selected_e[:-1]
    diff_idx = change_idx[1:] - change_idx[:-1]
    
    # Compare for the same sample and chop_vector index
    valid_diff_e = (diff_idx[:, 0] == 0) & (diff_idx[:, 1] == 0)
    diff_e *= valid_diff_e
    diff_e_neg = diff_e[diff_e < 0]

    return -np.sum(diff_e_neg) / mask.sum()