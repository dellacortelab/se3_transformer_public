import numpy as np
import os
import pickle as pkl
from ani1x_dataset import ANI1xDataset

test_dataset = ANI1xDataset(mode='test')

def read_train_losses(directory):
    energy_list = []
    force_list = []
        
    for j in range(1,1001):
        try:
            energy_file = f'energy_losses_{j}000.npy'
            energy_path = os.path.join(directory, energy_file)
            energy_loss = np.sqrt(np.mean(np.load(energy_path)**2))
            energy_list.append(energy_loss)

            force_file = f'force_losses_{j}000.npy'
            force_path = os.path.join(directory, force_file)
            force_loss = np.sqrt(np.mean(np.load(force_path)**2))
            force_list.append(force_loss)
        except:
            pass
    
    x_array = np.arange(1, 1001)*1000
    energy_array = np.array(energy_list)
    force_array = np.array(force_list)
        
    return x_array, energy_array, force_array

def read_val_losses(directory):
    with open(f'{directory}/validation.pkl','rb') as f:
        data = pkl.load(f)
    x_array = np.array(list(data['energy'].keys())[:100])
    energy_array = np.array(list(data['energy'].values())[:100])
    force_array = np.array(list(data['force'].values())[:100])
    
    return x_array, energy_array, force_array

def read_test_losses(directory, dataset=test_dataset, idx=1000000):
    with open(f'{directory}/test_{idx}.pkl','rb') as f:
        test_data = pkl.load(f)
    energy_error = 627.5 * np.sqrt(np.mean([(pred-true)**2 for pred, true in zip(test_data['energy_pred'], test_dataset.energy_list)]))
    force_error = 627.5 * np.sqrt(np.mean([np.mean((pred-true)**2) for pred, true in zip(test_data['force_pred'], test_dataset.force_list)]))
    return energy_error, force_error

def read_ensemble_data(model_dir, dataset, idx_list):
    dict_list = []
    for idx in idx_list:
        with open(f'{model_dir}/test_{idx}.pkl','rb') as f:
            data = pkl.load(f)
        dict_list.append(data)
    return dict_list

def ensemble_test_loss(dict_list, dataset=test_dataset, idx=1000000):
    force_pred = []
    energy_pred = []
    for i in range(len(test_dataset)):
        force_pred.append(np.mean([model_dict['force_pred'][i] for model_dict in dict_list], axis=0))
        energy_pred.append(np.mean([model_dict['energy_pred'][i] for model_dict in dict_list], axis=0))
    energy_error = 627.5 * np.sqrt(np.mean([(pred-true)**2 for pred, true in zip(energy_pred, test_dataset.energy_list)], axis=0))
    force_error = 627.5 * np.sqrt(np.mean([np.mean((pred-true)**2) for pred, true in zip(force_pred, test_dataset.force_list)], axis=0))
    return energy_error, force_error

plot_data = dict()
last_ten = np.arange(991000, 1000001, 1000)
model_name_list = ['basic', 'mean', 'deeper', 'wider', 'higher', 'heads']

for i in range(6):
    model_name = model_name_list[i]
    model_dir = f'./models/{model_name}'
    print(f'Reading data for {model_name} model')
    model_dict = {'train': dict(),
                  'val': dict(),
                  'test': dict()}
    
    # Train data
    x_train, energy_train, force_train = read_train_losses(model_dir)
    model_dict['train']['x'] = x_train
    model_dict['train']['energy'] = energy_train
    model_dict['train']['force'] = force_train
    
    # Validation data
    x_val, energy_val, force_val = read_val_losses(model_dir)
    model_dict['val']['x'] = x_val
    model_dict['val']['energy'] = energy_val
    model_dict['val']['force'] = force_val

    # Test data
    dict_list = read_ensemble_data(model_dir, test_dataset, last_ten)
    test_energy_error, test_force_error = losses = ensemble_test_loss(dict_list)
    model_dict['test']['energy'] = test_energy_error
    model_dict['test']['force'] = test_force_error
    
    plot_data[model_name] = model_dict
    
print('Saving data in pickle file')                                                 
with open('plot_data.pkl', 'wb') as f:
    pkl.dump(plot_data, f)
