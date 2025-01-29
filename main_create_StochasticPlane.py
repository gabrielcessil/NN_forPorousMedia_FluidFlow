from Utilities import domain_creator as dc
from Utilities.plotter import plot_heatmap
import numpy as np 
from Utilities import LBPM_runner
import json

# CREATING SEVERAL DOMAINS FOR LBPM
"""
# Nome base para salvar
simulation_base_name = "Example_Stoch"

# Parâmetros do meio poroso
N_Domains = 5
shape = (50, 50)  # Tamanho da imagem (altura, largura)
path_width = 5  # Largura dos caminhos (número de células)
n_paths = 2  # Número de caminhos
noise_level = 0.2  # Nível de ruído
tortuosity = 1.5 # Nível de tortuosidade
show = True
save_plot = True

# Load configs
with open("config.json" , "r") as json_file:
    config_loaded = json.load(json_file)
    
simulations_main_folder = config_loaded["LBPM_IO_folder"]

# Compute domain attributes
major_correlation_length = path_width  # Comprimento de correlação para criacao de caminhos principais
minor_correlation_length = max(path_width/2, 1)  # Comprimento de correlação para arredondamento e micro-estruturas

for i in range(N_Domains):
    
    # Create the domain and reflect
    midia_porosa = dc.create_media_fromTortuosity(shape, n_paths, path_width, noise_level, minor_correlation_length, tortuosity)
    midia_porosa_espelhada = dc.reflect_domain(midia_porosa)
    
    # Save it properly to LBPM
    simulation_name= simulation_base_name+"_"+str(i)
    
    if show: plot_heatmap(midia_porosa_espelhada, folder_name="", simulation_name="TESTE", save=False, show=show)
    
    LBPM_runner.Save_Example(midia_porosa_espelhada, simulation_name, simulations_main_folder, plot=save_plot)
"""


# EXAMPLE METHOD 1: Create media to match porosity
"""
# Parâmetros do meio poroso
porosity = 0.1
shape = (256, 256)  # Tamanho da imagem (altura, largura)
path_width = 10  # Largura dos caminhos (número de células)
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

"""
# EXAMPLE METHOD 2: Create media to match tortuosity
# Parâmetros do meio poroso
shape = (256, 256)  # Tamanho da imagem (altura, largura)
path_width = 10  # Largura dos caminhos (número de células)
n_paths = 2  # Número de caminhos
noise_level = 0.2  # Nível de ruído
tortuosity = 1.5
major_correlation_length = path_width  # Comprimento de correlação para criacao de caminhos principais
minor_correlation_length = max(path_width/2, 1)  # Comprimento de correlação para arredondamento e micro-estruturas
midia_porosa = dc.create_media_fromTortuosity(shape, n_paths, path_width, noise_level, minor_correlation_length, tortuosity)
plot_heatmap(midia_porosa, folder_name="", simulation_name="Stochastic_Example_2", show=True)
tortuosity_mean, tortuosidade_std = dc.calcula_tortuosidade(midia_porosa)
print(f"Tortuosidade desejada: {tortuosity}\nTortuosidade resultante: {tortuosity_mean} +\-{tortuosidade_std*3} \n")
midia_porosa_espelhada = dc.reflect_domain(midia_porosa)
plot_heatmap(midia_porosa_espelhada, folder_name="", simulation_name="Stochastic_Example_2", show=True)
midia_porosa_espelhada = midia_porosa_espelhada.astype(np.uint8)
"""




# EXAMPLE METHOD 3:
# Parâmetros do meio poroso
shape = (50, 50)  # Tamanho da imagem (altura, largura)
path_width = 4  # Largura dos caminhos (número de células)
porosity = 0.2
n_paths = 2  # Número de caminhos
tortuosity = 1.5
major_correlation_length = path_width  # Comprimento de correlação para criacao de caminhos principais
minor_correlation_length = max(path_width/2, 1)  # Comprimento de correlação para arredondamento e micro-estruturas
midia_porosa, midia_original = dc.combined_method(shape, n_paths, path_width, minor_correlation_length, tortuosity, porosity)
plot_heatmap(midia_porosa, folder_name="", simulation_name="Stochastic_Example_2", show=True)
plot_heatmap(midia_original, folder_name="", simulation_name="Stochastic_Example_2", show=True)
tortuosity_mean, tortuosidade_std = dc.calcula_tortuosidade(midia_porosa)
print(f"Tortuosidade desejada: {tortuosity}\nTortuosidade resultante: {tortuosity_mean} +\-{tortuosidade_std*3} \n")
midia_porosa_espelhada = dc.reflect_domain(midia_porosa)
plot_heatmap(midia_porosa_espelhada, folder_name="", simulation_name="Stochastic_Example_2", show=True)
midia_porosa_espelhada = midia_porosa_espelhada.astype(np.uint8)
porosidade_result = dc.calcula_porosidade(midia_porosa)
print(f"Porosidade desejada: {porosity}\nPorosidade resultante: {porosidade_result}\n")