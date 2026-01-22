import os
import ReadRecord
import numpy as np
import glob
import random
import Experiment_230508 as Experiment
import matplotlib.pyplot as plt
from backend import linear_protocol


def generate_hysteresis(EQ_data_dir,
                        hysteresis_data_dir,
                        target_data,
                        change_at2_to_numpy,
                        seed,
                        gm_scale_factor,
                        n_samples,
                        draw_hysteresis,
                        mat_type,
                        mat_props,
                        title,
                        EQ_list=False):
    if EQ_list == False:
        print('Making new EQ.AT2 files in {}'.format(EQ_data_dir))
        EQ_path = os.path.normpath(os.path.join(EQ_data_dir, './**/*.AT2'))
        EQ_list = glob.glob(EQ_path, recursive=True)
        random.seed(seed)
    
        EQ_list = random.sample(EQ_list, n_samples)
        random.shuffle(EQ_list)


    input_data = np.loadtxt(target_data, delimiter=' ')
    t = input_data[:, 0]
    gm = input_data[:, 1]
    dt = t[1]-t[0]

    max_value = abs(-9.8*gm*gm_scale_factor).max()
    print(max_value, 'm/s**2 is the scaled PGA for all samples')


    """Changing the EQ.AT2 file to .npy file"""
    dt_list, nPts_list = [], []
    for EQ_name in EQ_list:
        file_name, file_extension = os.path.splitext(EQ_name)
        dt, nPts = ReadRecord.ReadRecord(EQ_name, file_name+'.dat')
        dt_list.append(dt)
        nPts_list.append(nPts)
        gm = []
        with open(file_name+'.dat', 'r') as f:
            for line in f:
                if line:
                    words = line.split()
                    for word in words:
                        gm.append(float(word))
        gm = np.array(gm)*9.8
        scale_factor = max_value/abs(gm).max()
        gm = gm*scale_factor
        if change_at2_to_numpy:
            np.save(file_name+'.npy', gm)
            print('ground motions converted to numpy data')

    """Conduct analysis and save hysteresis data"""
    os.makedirs(hysteresis_data_dir, exist_ok=True)
    for i, EQ_name in enumerate(EQ_list):
        print("\r{}/{}".format(i+1, len(EQ_list)), end="")
        file_name, file_extension = os.path.splitext(EQ_name)
        gm = np.load(file_name+'.npy')
        # mat_prop = mat_props.copy()
        # mat_prop[0] = np.random.normal(loc=mat_props[0], scale=0.05*mat_props[0])
        t = np.arange(0, nPts_list[i]*dt_list[i], dt_list[i])
        outputs, beta_k = Experiment.dynamic_1DOF(mat_type, mat_props, t, gm/9.8, gm_scale=1)

        #### Calculate energy ####
        disp = outputs['rel_disp']
        force = outputs['force']
        ds = np.diff(disp, prepend=0)
        force_ = 1 / 2 * (np.concatenate(([0], force)) + np.concatenate((force, [0])))[:-1]
        energy = np.cumsum(force_ * ds)

        np.savez(os.path.normpath(os.path.join(hysteresis_data_dir, title + '_{}.npz'.format(i))), disp=outputs['rel_disp'],
                 vel=outputs['rel_vel'], force=outputs['force'], energy=energy)
        if draw_hysteresis:
            fig, ax = plt.subplots(figsize=(8,8))
            ax.plot(outputs['rel_disp'], outputs['force'])
            ax.set_xlabel('Displacement (m)')
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.grid()
            ax.set_title('{}'.format(i))
            fig.savefig(os.path.normpath(os.path.join(hysteresis_data_dir, './' + title + '_{}.png'.format(i))))
            plt.close(fig)

    np.savez(os.path.normpath(os.path.join(hysteresis_data_dir, './' + title + '_meta.npz')), dt_list=dt_list, nPts_list=nPts_list)
    print('Hysteresis data saved to {}'.format(hysteresis_data_dir))


def generate_impact_response(impact_length_list,
                             impact_magnitude_list,
                             hysteresis_data_dir,
                             draw_hysteresis,
                             mat_type,
                             mat_props,
                             title):
    raise NotImplementedError('This function is not implemented yet. Please use the following code snippet to generate impact response data')
    hysteresis_data_dir_impact = os.path.normpath(os.path.join(hysteresis_data_dir, './Impact'))
    os.makedirs(hysteresis_data_dir_impact, exist_ok=True)

    for impact_length in impact_length_list:
        for impact_magnitude in impact_magnitude_list:

            t = np.arange(0, impact_length)
            # impact_loading for sine impact
            # impact_loading = impact_magnitude * np.sin(np.pi * t / impact_duration)

            # impact_loading for sharp impact
            # impact_loading = np.concatenate((np.linspace(0, impact_magnitude, num=len(t)//2),
            #       np.linspace(impact_magnitude, 0, num=len(t)//2)))

            # impact_loading for smooth impact
            disp = impact_magnitude * np.exp(-((t-max(t)/2)**2) / (2 * (impact_length / 10)**2)) - impact_magnitude * np.exp(-((0-max(t)/2)**2) / (2 * (impact_length / 10)**2))
            
            # Add pre-impact loading (zero - process)
            t = np.concatenate((np.arange(0, 1), t + 1))
            disp = np.concatenate((np.zeros((len(np.arange(0, 1)))), disp))

            outputs = Experiment.static_1DOF(mat_type, mat_props, disp)

            #### Calculate energy ####
            disp = outputs['disp']
            force = outputs['force']
            ds = np.diff(disp, prepend=0)
            force_ = 1 / 2 * (np.concatenate(([0], force)) + np.concatenate((force, [0])))[:-1]
            energy = np.cumsum(force_ * ds)

            np.savez(os.path.normpath(os.path.join(hysteresis_data_dir_impact, title + '_{}_{}.npz'.format(impact_length, impact_magnitude))), disp=outputs['disp'],
                        force=outputs['force'], energy=energy)
            if draw_hysteresis:
                fig, ax = plt.subplots(figsize=(8,8))
                ax.plot(outputs['disp'], outputs['force'])
                ax.set_xlabel('Displacement (m)')
                ax.tick_params(axis='both', which='major', labelsize=15)
                ax.grid()
                ax.set_title('{}_{}'.format(impact_length, impact_magnitude))
                fig.savefig(os.path.normpath(os.path.join(hysteresis_data_dir_impact, './' + title + '_{}_{}.png'.format(impact_length, impact_magnitude))))
                plt.close(fig)


    print('Hysteresis data saved to {}'.format(hysteresis_data_dir_impact))


def generate_linear_protocol(hysteresis_data_dir_linear_protocol,
                                draw_hysteresis,
                                mat_type,
                                mat_props,
                                title,
                                slopes,
                                intercepts,
                                periods,
                                repetitions):
    
        
        os.makedirs(hysteresis_data_dir_linear_protocol, exist_ok=True)
        i = 0
        for period, repetition in zip(periods, repetitions):
            for slope, intercept in zip(slopes, intercepts):
                disp = linear_protocol(slope, intercept, period, repetition)
                outputs = Experiment.static_1DOF(mat_type, mat_props, disp)

                #### Calculate energy ####
                disp = outputs['disp']
                force = outputs['force']
                ds = np.diff(disp, prepend=0)
                force_ = 1 / 2 * (np.concatenate(([0], force)) + np.concatenate((force, [0])))[:-1]
                energy = np.cumsum(force_ * ds)

                np.savez(os.path.normpath(os.path.join(hysteresis_data_dir_linear_protocol, title + '_{}.npz'.format(i))), disp=outputs['disp'],
                            force=outputs['force'], energy=energy)
                
                if draw_hysteresis:
                    fig, ax = plt.subplots(figsize=(8,8))
                    ax.plot(outputs['disp'], outputs['force'])
                    ax.set_xlabel('Displacement (m)')
                    ax.tick_params(axis='both', which='major', labelsize=15)
                    ax.grid()
                    ax.set_title('{}_{}'.format(period, slope))
                    fig.savefig(os.path.normpath(os.path.join(hysteresis_data_dir_linear_protocol, './' + title + '_{}.png'.format(i))))
                    plt.close(fig)

                i += 1
    
        print('Hysteresis data saved to {}'.format(hysteresis_data_dir_linear_protocol))


def validate_data(EQ_list, 
                  hysteresis_data_dir, 
                  title,
                  mat_type,
                  mat_props,
                    draw_hysteresis):
    
    hysteresis_data_dir_validation = os.path.normpath(os.path.join(hysteresis_data_dir, './Validation'))

    os.makedirs(hysteresis_data_dir_validation, exist_ok=True)

    for i, EQ_name in enumerate(EQ_list):
        data_transient = np.load(os.path.normpath(os.path.join(hysteresis_data_dir, title + '_{}.npz'.format(i))))
        disp = data_transient['disp']
        force = data_transient['force']
        data_static = Experiment.static_1DOF(mat_type, mat_props, disp)

        if draw_hysteresis:
            fig, ax = plt.subplots(figsize=(8,8))
            ax.plot(disp, force)
            ax.plot(data_static['disp'], data_static['force'])
            ax.set_xlabel('Displacement (m)')
            ax.set_ylabel('Force (N)')
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.grid()
            fig.savefig(os.path.normpath(os.path.join(hysteresis_data_dir_validation, './' + title + '_{}.png'.format(i))))
            plt.close(fig)

def validate_data_impact(title,
                         mat_type,
                         mat_props,
                         hysteresis_data_dir,
                         draw_hysteresis):
    hysteresis_data_dir_impact = os.path.normpath(os.path.join(hysteresis_data_dir, './Impact'))
    hysteresis_data_dir_validation = os.path.normpath(os.path.join(hysteresis_data_dir_impact, './Validation'))
    os.makedirs(hysteresis_data_dir_validation, exist_ok=True)

    filepaths = glob.glob(os.path.join(hysteresis_data_dir_impact, title + '_*.npz'))

    for i, filepath in enumerate(filepaths):
        data_transient = np.load(filepath)
        disp = data_transient['disp']
        force = data_transient['force']
        data_static = Experiment.static_1DOF(mat_type, mat_props, disp)

        if draw_hysteresis:
            fig, ax = plt.subplots(figsize=(8,8))
            ax.plot(disp, force)
            ax.plot(data_static['disp'], data_static['force'])
            ax.set_xlabel('Displacement (m)')
            ax.set_ylabel('Force (N)')
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.grid()
            filename, extension = os.path.splitext(os.path.basename(filepath))
            fig.savefig(os.path.normpath(os.path.join(hysteresis_data_dir_validation, './' + filename + '.png'.format(i))))
            plt.close(fig)
