from hdf5storage import loadmat
from Utilities.plotter import plot_heatmap, Plot_Domain
import numpy as np
import os

def dividir_imagem_em_blocos(imagem, shape):
    """
    Divide uma imagem em blocos de tamanho especificado.

    Args:
        imagem (np.ndarray): Array 2D ou 3D representando a imagem.
        shape (tuple): Shape (altura, largura) dos blocos.

    Returns:
        list: Lista de sub-blocos (arrays 2D ou 3D).
    """
    altura, largura = imagem.shape[:2]  # Dimensões da imagem
    bloco_altura, bloco_largura = shape  # Dimensões do bloco desejado
    
    # Validar se o shape dos blocos é válido
    if bloco_altura > altura or bloco_largura > largura:
        raise ValueError("As dimensões dos blocos são maiores que as da imagem.")
    
    sub_blocos = []
    
    # Percorrer a imagem em blocos
    for i in range(0, altura - bloco_altura + 1, bloco_altura):
        for j in range(0, largura - bloco_largura + 1, bloco_largura):
            # Extrair o sub-bloco usando slicing
            sub_bloco = imagem[i:i + bloco_altura, j:j + bloco_largura]
            sub_blocos.append(sub_bloco)
    
    return sub_blocos

def dividir_array_3d(array_3d, axis, n_planos, shape):
    """
    Divide um array 3D em N planos 2D ao longo de um eixo e divide cada plano em sub-planos.

    Args:
        array_3d: O array 3D a ser dividido.
        axis: O eixo ao longo do qual a divisão será feita (0 para X, 1 para Y, 2 para Z).
        n_planos: O número de planos a serem criados.
        shape: A forma desejada para os sub-planos (tupla de inteiros).

    Returns:
        Uma lista de listas, onde cada lista interna contém os sub-planos de um plano.
    """
    # Verificando se o número de planos é válido
    if n_planos > array_3d.shape[axis]:
        raise ValueError("Número de planos maior que o tamanho do eixo.")

    # Calculando o tamanho de cada plano
    step = array_3d.shape[axis] // n_planos

    # Lista para armazenar os sub-planos de todos os planos
    sub_planos = []

    for i in range(n_planos):
        # Extraindo o plano ao longo do eixo especificado
        plano_index = i * step

        if axis == 0:
            plano = array_3d[plano_index, :, :]
        elif axis == 1:
            plano = array_3d[:, plano_index, :]
        elif axis == 2:
            plano = array_3d[:, :,plano_index]
        else:
            raise ValueError("O eixo deve ser 0, 1 ou 2.")
        
        # Remove dimensao==1, tornando array 3D em 2D
        plano = np.squeeze(plano)
        
        # Lista para armazenar os sub-planos do plano atual
        sub_planos.extend(dividir_imagem_em_blocos(plano, shape=shape))
        

    return sub_planos
    
fluid_value = 1
rocks_directory= "./Rocks/"
planes_directory= "./Planes/"
plots_directory=  "/media/gabriel/a91864d5-f0d3-4994-921d-b2238c6ce60e/labcc/Área de Trabalho/Gabriel/"
sub_shape=(50,50)
all_samples = {}
arquivos_mat = [arquivo for arquivo in os.listdir(rocks_directory) if arquivo.endswith(".mat")]
for arq in arquivos_mat:
    
    archieve = rocks_directory+arq
    media_rock = loadmat(archieve)['bin']
    rock_name = arq.rstrip('.mat')
    
    Plot_Domain(media_rock, filename=plots_directory+rock_name+"_rock", remove_value=[fluid_value])
    
    for axis in [0,1,2]:
        n_planos = media_rock.shape[axis]//10
        
        sub_rocks_planes = dividir_array_3d(media_rock, axis, n_planos, sub_shape)
        
        for i,plane in enumerate(sub_rocks_planes):
            all_samples[rock_name+f"_{axis}_{i}"] = plane
        

for domain_name,domain in all_samples.items(): 
    plot_heatmap(domain, filename=plots_directory+domain_name,vmin=0,vmax=1)

    domain.tofile(planes_directory+domain_name+".raw")