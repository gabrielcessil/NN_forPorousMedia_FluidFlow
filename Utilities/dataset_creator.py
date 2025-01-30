import torch
from torch.utils.data import Dataset, random_split
from hdf5storage import loadmat
import numpy as np
from torch.utils.data import DataLoader

# CREATING DATASET

### COLETA DADOS - ROCHA E SIMULACAO
# 3D Geometry Files
# Source: https://www.digitalrocksportal.org/projects/372
# Load binary, simulation results, and features (.mat) as 3D Numpy array

class Data_from_Simulation():
    
    def __init__(self, file_paths, save_name=None, subshape=(50,50,50)):
        self.data = []


        for archieve in file_paths:
            # Load data
            media = loadmat(archieve)
            media_rock, media_ux, media_uy = media['rho'],  media['ux'],  media['uy']
            
            
            
            # Divide 3D domain into sub 3D with smaller 'subshape'
            sub_medias = self.slice_multiple_arrays_together(
                arrays=[media_rock,media_ux,media_uy],
                subshape=subshape
            )
            
            
            
            # Colete 2D planes from each sub_media
            for sub_media in sub_medias:
                media_rock_1, media_rock_2 = self.coletar_face_e_plano_central(sub_media[0], face="xy")
                media_ux_1, media_ux_2 =     self.coletar_face_e_plano_central(sub_media[1], face="xy")
                media_uy_1, media_uy_2 =     self.coletar_face_e_plano_central(sub_media[2], face="xy")
            
                self.data.append([media_rock_1, media_ux_1, media_uy_1])
                self.data.append([media_rock_2, media_ux_2, media_uy_2])
        
        # Save data
        if save_name is not None: torch.save(self.data, save_name+".pt")
    
        

    def coletar_face_e_plano_central(self, values, face="xy"):
        """
        Coleta a face do cubo na direção especificada (xy, yz, xz) e o plano central dessa direção.
    
        Parâmetros:
            values (np.ndarray): Matriz 3D de valores.
            face (str): Direção da face a ser coletada ('xy', 'yz', 'xz').
    
        Retorna:
            face_pos (np.ndarray): A face do cubo na direção especificada.
            plano_central (np.ndarray): O plano central na direção especificada.
        """
        if face == "xy":
            # Face xy: seleciona todas as camadas Z e as duas dimensões X, Y
            face_pos = values[:, :, 0]  # Exemplo de face na direção XY (plano Z=0)
            plano_central = values[values.shape[0] // 2, values.shape[1] // 2, :]  # No entanto, queremos uma fatia 2D
    
            # Seleciona o plano central na direção XY
            plano_central = values[values.shape[0] // 2, :, :]  # Fatia 2D ao longo de Z
    
        elif face == "yz":
            # Face yz: seleciona todas as camadas X e as duas dimensões Y, Z
            face_pos = values[0, :, :]  # Exemplo de face na direção YZ (plano X=0)
            plano_central = values[:, values.shape[1] // 2, :]  # Fatia 2D ao longo de X
    
        elif face == "xz":
            # Face xz: seleciona todas as camadas Y e as duas dimensões X, Z
            face_pos = values[:, 0, :]  # Exemplo de face na direção XZ (plano Y=0)
            plano_central = values[:, :, values.shape[2] // 2]  # Fatia 2D ao longo de Y
        
        else:
            raise ValueError("Direção inválida. Use 'xy', 'yz' ou 'xz'.")
        
        return face_pos, plano_central
    
    def slice_multiple_arrays_together(self, arrays, subshape):
        """
        Fatia múltiplos arrays 3D juntos, retornando subamostras alinhadas.
        
        Parameters:
        -----------
        arrays : list of np.ndarray
            Lista de arrays 3D que serão fatiados juntos.
        subshape : tuple of ints
            Tamanho das subamostras 3D (profundidade, altura, largura).
            
        Returns:
        --------
        list of list of np.ndarray
            Lista de listas, onde cada sublista contém as subamostras dos arrays
            na mesma posição.
        """
        # Certifique-se de que todos os arrays têm a mesma forma
        array_shapes = [array.shape for array in arrays]
        if not all(shape == array_shapes[0] for shape in array_shapes):
            raise ValueError("Todos os arrays devem ter o mesmo formato.")
    
        # Dimensão do primeiro array para criar os bins
        shape = arrays[0].shape
        z_bins = np.arange(0, (shape[0] // subshape[0]) * subshape[0] + 1, subshape[0])
        y_bins = np.arange(0, (shape[1] // subshape[1]) * subshape[1] + 1, subshape[1])
        x_bins = np.arange(0, (shape[2] // subshape[2]) * subshape[2] + 1, subshape[2])
    
        # Coletar subamostras alinhadas
        grouped_subsamples = []
        for z_start, z_end in zip(z_bins[:-1], z_bins[1:]):
            for y_start, y_end in zip(y_bins[:-1], y_bins[1:]):
                for x_start, x_end in zip(x_bins[:-1], x_bins[1:]):
                    # Coletar subamostras alinhadas para todos os arrays na mesma posição
                    subsamples = [
                        array[z_start:z_end, y_start:y_end, x_start:x_end]
                        for array in arrays
                    ]
                    grouped_subsamples.append(subsamples)
    
        return tuple(grouped_subsamples)


### LOADING DATASET 
def array_type_check(variable):
    if isinstance(variable, list):  # Check if it's a list
        return np.array(variable)  # Convert to numpy array
    elif isinstance(variable, np.ndarray):  # Check if it's already an ndarray
        return variable  # Return as is
    else:
        raise TypeError("Dataset samples must be a list or a numpy ndarray")
        
def check_shape_array(array, expected_shape):
    # Check if the array is a numpy ndarray
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a numpy ndarray")

    # Check the current shape of the array
    current_shape = array.shape

    # If the current shape is not as expected, try reshaping
    if current_shape != expected_shape:
        try:
            reshaped_array = array.reshape(expected_shape)
            return reshaped_array
        except ValueError:
            raise ValueError(f"Unexpected shape {current_shape} that cannot reshape to {expected_shape}")
    
    # Return the array if it already has the expected shape
    return array

def get_input_output_tensors(file_path, array_shape=(50,50)):
    """
    Processa os dados carregados e converte em tensores apropriados.

    Args:
        file_path (str): Caminho para o arquivo .pt contendo os dados.
        array_shape (tuple): Formato desejado para as imagens (ex: (50, 50)).

    Retorna:
        input_tensor_dataset (list): Lista de tensores de entrada (imagens).
        output_tensors_dataset (list): Lista de tensores de saída.
    """
    loaded_dataset = torch.load(file_path)

    input_tensor_dataset = []
    output_tensors_dataset = []

    for group in loaded_dataset:
        media_rock, media_ux, media_uy = group[0], group[1], group[2]
        
        # Convertendo para o tipo correto (ex: NumPy -> tensor)
        media_rock = array_type_check(media_rock)
        media_ux = array_type_check(media_ux)
        media_uy = array_type_check(media_uy)

        # Garantir o formato correto
        media_rock = check_shape_array(media_rock, array_shape)
        media_ux = check_shape_array(media_ux, array_shape)
        media_uy = check_shape_array(media_uy, array_shape)

        # Converte para tensores PyTorch
        # Caso já seja um numpy array, basta apenas converter para tensor diretamente
        media_rock = torch.tensor(media_rock).float().unsqueeze(0)  # Adiciona a dimensão do canal para Conv2D

        # Para as saídas, mantemos os arrays achatados
        media_ux = torch.tensor(media_ux.flatten()).float()  # Flatten para saída Linear
        media_uy = torch.tensor(media_uy.flatten()).float()  # Flatten para saída Linear
        
        # Adicionando aos datasets
        input_tensor_dataset.append(media_rock)
        output_tensors_dataset.append(media_ux)

    return input_tensor_dataset, output_tensors_dataset
class TensorsDataset(Dataset):
    def __init__(self, inputs, outputs):
        
        self.inputs, self.outputs = inputs, outputs
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Return input and corresponding output tensors
        return self.inputs[idx], self.outputs[idx]  # Output shape: (2, H, W)

    def split(self, train_ratio=0.7, val_ratio=0.15):
        """
        Splits the dataset into training, validation, and test sets.

        Args:
            train_ratio (float): Proportion of the dataset to use for training (default: 0.7).
            val_ratio (float): Proportion of the dataset to use for validation (default: 0.15).

        Returns:
            train_dataset (Dataset): Training dataset.
            val_dataset (Dataset): Validation dataset.
            test_dataset (Dataset): Test dataset.
        """
        total_size = len(self)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size  # Remaining samples for test

        return random_split(self, [train_size, val_size, test_size])

    def get_dataloaders(self, train_ratio=0.7, val_ratio=0.15, batch_size=10):
        train_dataset, val_dataset, test_dataset = self.split(train_ratio=train_ratio, val_ratio=val_ratio) # Split into train, validation, and test sets
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # For trainin
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)  # For validation
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)  # For testing
        
        return train_loader, val_loader, test_loader
        
    
