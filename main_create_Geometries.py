from Utilities import domain_creator as dc
from Utilities.plotter import plot_heatmap
import numpy as np 
from Utilities import LBPM_runner
import json



#######################################################
#************ USER INPUTS: Hyperparameters ***********#
#######################################################

simulation_base_name = "Example_Tubes" # Nome base para salvar meios porosos
N_Domains = 30 # Numero de meios porosos criados
path_width = 40  # Largura dos caminhos (número de células)
n_paths = 1  # Número de caminhos
noise_level = 0.2  # Nível de ruído
tortuosity = 1.05 # Nível de tortuosidade
porosity = 0.5 # Nivel de tortuosidade
show = True
save_plot = True
reflect = True



#######################################################
#************ LOAD CONFIGS                 ***********#
#######################################################

# Load configs
with open("config.json" , "r") as json_file:
    config_loaded = json.load(json_file)
simulations_main_folder = config_loaded["LBPM_IO_folder"]
rock_shape = config_loaded["Rock_shape"]  # (D, H, W) or (C, D, H, W)




#######################################################
#************ COMPUTATIONS                 ***********#
####################################################### 


# Compute domain attributes
major_correlation_length = path_width  # Correlation length of the major paths
minor_correlation_length = max(path_width/2, 1)  # Correlation length of minor-structures round path's corners



for i in range(N_Domains):
    
    # Create the domain and reflect 
    rock_shape_2D = (rock_shape[-2], rock_shape[-1])
    midia_porosa = dc.create_media_fromTortuosity(rock_shape_2D, n_paths, path_width, minor_correlation_length, tortuosity)
    midia_porosa_espelhada = dc.reflect_domain(midia_porosa)
    
    # Save it properly to LBPM
    simulation_name= simulation_base_name+"_"+str(i)
    
    if show: plot_heatmap(midia_porosa_espelhada, save=False, show=show)
    
    LBPM_runner.Save_Example(midia_porosa_espelhada, simulation_name, simulations_main_folder, plot=save_plot)




#******************************************************************************#
#######################################################
#************ EXAMPLES OF EACH METHOD      ***********#
####################################################### 


# EXAMPLE METHOD 1: Create media to match porosity
"""
# Parâmetros do meio poroso
porosity = 0.1
shape = (100, 100)  # Tamanho da imagem (altura, largura)
path_width = 10  # Largura dos caminhos (número de células)
print("\nMethod 1: match porosity")
major_correlation_length = path_width  # Comprimento de correlação para criacao de caminhos principais
minor_correlation_length = max(path_width/2, 1)  # Comprimento de correlação para arredondamento e micro-estruturas
porous_media = dc.create_media_fromPorosity(shape, major_correlation_length, porosity)
plot_heatmap(porous_media, folder_name="", simulation_name="Stochastic_Example_1_Poros_01", show=True)
porosidade_result = dc.calcula_porosidade(porous_media)
print(f"Porosidade desejada: {porosity}\nPorosidade resultante: {porosidade_result}\n")
porosity = 0.3
porous_media = dc.create_media_fromPorosity(shape, major_correlation_length, porosity)
plot_heatmap(porous_media, folder_name="", simulation_name="Stochastic_Example_1_Poros_03", show=True)
porosidade_result = dc.calcula_porosidade(porous_media)
print(f"Porosidade desejada: {porosity}\nPorosidade resultante: {porosidade_result}\n")
porosity = 0.5
porous_media = dc.create_media_fromPorosity(shape, major_correlation_length, porosity)
plot_heatmap(porous_media, folder_name="", simulation_name="Stochastic_Example_1_Poros_05", show=True)
porosidade_result = dc.calcula_porosidade(porous_media)
print(f"Porosidade desejada: {porosity}\nPorosidade resultante: {porosidade_result}\n")
"""

# EXAMPLE METHOD 2: Create media to match tortuosity
"""
# Parâmetros do meio poroso
shape = (250, 250)  # Tamanho da imagem (altura, largura)
path_width = 40 # Largura dos caminhos (número de células)
n_paths = 1  # Número de caminhos
tortuosity = 1.05
print("\nMethod 2: match tortuosity with tubes")
major_correlation_length = path_width  # Comprimento de correlação para criacao de caminhos principais
minor_correlation_length = max(path_width/2, 1)  # Comprimento de correlação para arredondamento e micro-estruturas
midia_porosa = dc.create_media_fromTortuosity(shape, n_paths, path_width, minor_correlation_length, tortuosity)
plot_heatmap(midia_porosa, folder_name="", simulation_name="Stochastic_Example_2", show=True)
tortuosity_mean, tortuosidade_std = dc.calcula_tortuosidade(midia_porosa)
print(f"Tortuosidade desejada: {tortuosity}\nTortuosidade resultante: {tortuosity_mean} +\-{tortuosidade_std*3} \n")
"""

# EXAMPLE METHOD 3: Create media to match porosity with tubes enough, where tubes have certain tortuosity
"""
# Parâmetros do meio poroso
shape = (250, 250)  # Tamanho da imagem (altura, largura)
path_width = 3  # Largura dos caminhoporosos (número de células)
porosity = 0.3
tortuosity = 1.5
print("\nMethod 3: match porosity with tubes enough, where tubes have certain tortuosity")
major_correlation_length = path_width  # Comprimento de correlação para criacao de caminhos principais
minor_correlation_length = max(path_width/2, 1)  # Comprimento de correlação para arredondamento e micro-estruturas
midia_porosa = dc.create_media_fromTortuosity_and_porosity(shape, path_width, minor_correlation_length, tortuosity, porosity)
plot_heatmap(midia_porosa, folder_name="", simulation_name="Stochastic_Example_2", show=True)
tortuosity_mean, tortuosidade_std = dc.calcula_tortuosidade(midia_porosa)
porosidade_result = dc.calcula_porosidade(midia_porosa)
print(f"Tortuosidade desejada: {tortuosity}\nTortuosidade resultante: {tortuosity_mean} +\-{tortuosidade_std*3} \n")
print(f"Porosidade desejada: {porosity}\nPorosidade resultante: {porosidade_result}\n")
"""


# EXAMPLE METHOD 4: Create media to match tortuosity and porosity with noised tubes
"""
# Parâmetros do meio poroso
shape = (250, 250)  # Tamanho da imagem (altura, largura)
path_width = 6  # Largura dos caminhos (número de células)
porosity = 0.2
tortuosity = 1.5
print("\nMethod 4: match tortuosity and porosity with noised tubes")
major_correlation_length = path_width  # Comprimento de correlação para criacao de caminhos principais
minor_correlation_length = max(path_width/1.8, 1)  # Comprimento de correlação para arredondamento e micro-estruturas
midia_porosa, midia_original = dc.create_media_fromPorousTortuosity_and_porosity(shape, path_width, minor_correlation_length, tortuosity, porosity)
plot_heatmap(midia_porosa, folder_name="", simulation_name="Stochastic_Example_2", show=True)
plot_heatmap(midia_original, folder_name="", simulation_name="Stochastic_Example_2", show=True)
tortuosity_mean, tortuosidade_std = dc.calcula_tortuosidade(midia_porosa)
print(f"Tortuosidade desejada: {tortuosity}\nTortuosidade resultante: {tortuosity_mean} +\-{tortuosidade_std*3} \n")
porosidade_result = dc.calcula_porosidade(midia_porosa)
print(f"Porosidade desejada: {porosity}\nPorosidade resultante: {porosidade_result}\n")
"""
