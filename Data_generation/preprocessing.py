import numpy as np
import os
from sklearn.model_selection import train_test_split

def preprocess(n_samples,
               hysteresis_data_dir,
               processed_data_dir,
               title,
               val_size=False,
               normalize_gap=False,
               test=True):
    # Total energy as input
    X = []
    y = []

    time_length = []
    num_inputs = 2  # Displacement and mask
    for i in range(n_samples):
        outputs = np.load(os.path.normpath(os.path.join(hysteresis_data_dir, './' + title + '_{}.npz'.format(i))))
        disp = outputs['disp']
        force = outputs['force']
        energy = outputs['energy']
        X.append(np.concatenate((disp[:, np.newaxis], np.ones((len(disp), 1))), axis=1))
        y.append(force)
        time_length.append(len(disp))
    time_length = np.array(time_length)
    max_time_length = np.max(time_length)
    for i in range(n_samples):
        if len(X[i]) < max_time_length:
            X[i] = np.concatenate((X[i], np.zeros((max_time_length - len(X[i]), num_inputs))), axis=0)
            y[i] = np.concatenate((y[i], np.zeros((max_time_length - len(y[i])))), axis=0)
    X, y = np.array(X), np.array(y)    # X.shape = (n_samples, max_time_length, num_inputs), y.shape = (n_samples, max_time_length)
    X_max, y_max = None, None
    
    if normalize_gap != False:
        X_max = np.max(np.abs(X), axis=(0,1))[:num_inputs-1]
        y_max = np.max(np.abs(y))
        X[:, :, :num_inputs - 1] = X[:, :, :num_inputs - 1] / (X_max * (1 + normalize_gap))
        y = y / (y_max * (1 + normalize_gap))

    if test:
        X_train, X_test, y_train, y_test = X[:int(n_samples*0.5)], X[int(n_samples*0.5):], y[:int(n_samples*0.5)], y[int(n_samples*0.5):]

    if val_size != False:
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_size, random_state=0)

    if test:
        np.savez(os.path.normpath(os.path.join(processed_data_dir, './' + title + '_Processed_data.npz')), X_train=X_train, X_test=X_test, X_val=X_val, 
                y_train=y_train, y_test=y_test, y_val=y_val, normalize_gap=normalize_gap, X_max=X_max, y_max=y_max)
    else:
        np.savez(os.path.normpath(os.path.join(processed_data_dir, './' + title + '_Processed_data.npz')), X_train=X_train, X_val=X_val, 
                y_train=y_train, y_val=y_val, normalize_gap=normalize_gap, X_max=X_max, y_max=y_max)

def preprocess_loading_protocol(n_samples,
                                hysteresis_data_dir,
                                processed_data_dir,
                                title,
                                val_size=False,
                                normalize_gap=False,
                                X_max=False,
                                y_max=False,
                                test=True,
                                data_augmentation_rates=False):
    # Total energy as input
    X = []
    y = []

    time_length = []
    num_inputs = 2
    for i in range(n_samples):
        outputs = np.load(os.path.normpath(os.path.join(hysteresis_data_dir, './' + title + '_{}.npz'.format(i))))
        disp = outputs['disp']
        force = outputs['force']
        energy = outputs['energy']
        X.append(np.concatenate((disp[:, np.newaxis], np.ones((len(disp), 1))), axis=1))
        y.append(force)
        time_length.append(len(disp))
    time_length = np.array(time_length)
    max_time_length = np.max(time_length)
    for i in range(n_samples):
        if len(X[i]) < max_time_length:
            X[i] = np.concatenate((X[i], np.zeros((max_time_length - len(X[i]), num_inputs))), axis=0)
            y[i] = np.concatenate((y[i], np.zeros((max_time_length - len(y[i])))), axis=0)
    X, y = np.array(X), np.array(y)

    raise NotImplementedError("This part is not implemented yet.")

    if normalize_gap != False:
        if (X_max == False).all() or y_max == False:
            X_max = np.max(np.abs(X), axis=(0,1))[:2]
            y_max = np.max(np.abs(y))
        X[:, :, :2] = X[:, :, :2] / (X_max * (1 + normalize_gap))
        y = y / (y_max * (1 + normalize_gap))

    if test:
        X_train, X_test, y_train, y_test = X[:int(n_samples*0.5)], X[int(n_samples*0.5):], y[:int(n_samples*0.5)], y[int(n_samples*0.5):]
    else:
        X_train, y_train = X, y

    if val_size != False:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=0)

    if data_augmentation_rates != False:
        augmented_data = []
        for data_augmentation_rate in data_augmentation_rates:
            X_train_i = X_train[:, ::data_augmentation_rate, :]
            X_train_i = np.concatenate((X_train_i, np.zeros((len(X_train_i), max_time_length - X_train_i.shape[1], num_inputs))), axis=1)
            y_train_i = y_train[:, ::data_augmentation_rate]
            y_train_i = np.concatenate((y_train_i, np.zeros((len(y_train_i), max_time_length - y_train_i.shape[1]))), axis=1)
            augmented_data.append((X_train_i, y_train_i))
        
        for i in range(len(data_augmentation_rates)):
            X_train = np.concatenate((X_train, augmented_data[i][0]), axis=0)
            y_train = np.concatenate((y_train, augmented_data[i][1]), axis=0)

    if test:
        np.savez(os.path.normpath(os.path.join(processed_data_dir, './' + title + '_Processed_data.npz')), X_train=X_train, X_test=X_test, X_val=X_val, 
                y_train=y_train, y_test=y_test, y_val=y_val, normalize_gap=normalize_gap, X_max=X_max, y_max=y_max)
    else:
        np.savez(os.path.normpath(os.path.join(processed_data_dir, './' + title + '_Processed_data.npz')), X_train=X_train, X_val=X_val, 
                y_train=y_train, y_val=y_val, normalize_gap=normalize_gap, X_max=X_max, y_max=y_max)



def preprocess_impact_loading(impact_length_list,
                             impact_magnitude_list,
                              hysteresis_data_dir,
                              processed_data_dir,
                              title):
    
    raise NotImplementedError("This part is not implemented yet.")
    processed_data_0 = np.load(os.path.normpath(os.path.join(processed_data_dir, './' + title + '_Processed_data.npz')))
    processed_data_dir_impact = os.path.normpath(os.path.join(processed_data_dir, './Impact'))

    os.makedirs(processed_data_dir_impact, exist_ok=True)
    normalize_gap = processed_data_0['normalize_gap']
    X_max = processed_data_0['X_max']
    y_max = processed_data_0['y_max']
    # Total energy as input
    X = []
    y = []

    time_length = []
    num_inputs = 2 + 1
    hysteresis_data_dir_impact = os.path.normpath(os.path.join(hysteresis_data_dir, './Impact'))
    i = 0
    for impact_length in impact_length_list:
        for impact_magnitude in impact_magnitude_list:
            outputs = np.load(os.path.normpath(os.path.join(hysteresis_data_dir_impact, './' + title + '_{}_{}.npz'.format(impact_length, impact_magnitude))))
            disp = outputs['disp']
            diff_disp = np.diff(disp, prepend=0)
            force = outputs['force']
            energy = outputs['energy']
            X.append(np.concatenate((disp[:, np.newaxis], diff_disp[:, np.newaxis], np.ones((len(disp), 1))), axis=1))
            y.append(force)
            time_length.append(len(disp))
            i += 1

    time_length = np.array(time_length)
    max_time_length = np.max(time_length)
    i = 0
    for impact_length in impact_length_list:
        for impact_magnitude in impact_magnitude_list:

            if len(X[i]) < max_time_length:
                X[i] = np.concatenate((X[i], np.zeros((max_time_length - len(X[i]), num_inputs))), axis=0)
                y[i] = np.concatenate((y[i], np.zeros((max_time_length - len(y[i])))), axis=0)
            i += 1
                
    X, y = np.array(X), np.array(y)

    X[:, :, :2] = X[:, :, :2] / (X_max * (1 + normalize_gap))
    y = y / (y_max * (1 + normalize_gap))

    np.savez(os.path.normpath(os.path.join(processed_data_dir_impact, './' + title + '_Processed_data_impact.npz')), X_test=X, y_test=y)