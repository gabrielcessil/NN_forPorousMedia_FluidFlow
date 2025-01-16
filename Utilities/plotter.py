import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as py

def Plot_Domain(values, filename, remove_value=[]):
    """
    Plot a 3D domain from a 3D NumPy array and ghost cells with a specific value.

    Parameters:
        values (np.ndarray): 3D NumPy array of cell values.
        filename (str): Name of the output file (without extension).
        remove_value (float/int): Value to mark as ghost cells.
    """
    # Criar grid estruturado (ImageData)
    grid = pv.ImageData()
    grid.dimensions = np.array(values.shape) + 1  # Dimensões como pontos
    grid.origin = (0, 0, 0)  # Origem da grade
    grid.spacing = (1, 1, 1)  # Espaçamento uniforme

    # Atribuir valores aos dados das células
    grid.cell_data["values"] = values.flatten(order="F")  # Nome do atributo: "values"

    # Remove celulas indesejadas no plot
    mesh = grid.cast_to_unstructured_grid()
    if remove_value:
        for removed_value in remove_value:
            ghosts = np.argwhere(mesh["values"] == removed_value)
            mesh.remove_cells(ghosts.flatten(), inplace=True)

    # Plot the grid with ghosted cells hidden
    plotter = pv.Plotter(window_size=[1920, 1080], off_screen=True)  # Full HD resolution

    # Adiciona celulas
    plotter.add_mesh(
        mesh,
        cmap="viridis",
        show_edges=False,
        lighting=True,
        smooth_shading=True,
        split_sharp_edges=True,
        scalar_bar_args={
            "title": "Range",          # Title of the color bar
            "vertical": True,          # Make the color bar vertical
            "title_font_size": 20,
            "label_font_size": 16,
            "position_x": 0.85,        # Position of the color bar (X-axis)
            "position_y": 0.05,        # Position of the color bar (Y-axis)
            "height": 0.9,             # Height of the color bar
            "width": 0.05              # Width of the color bar
        }
    )

    # Adiciona linhas nas arestas
    edges = mesh.extract_feature_edges(
        boundary_edges=False,
        non_manifold_edges=False,
        feature_angle=30,
        manifold_edges=False,
    )
    plotter.add_mesh(edges,
                     color='k',
                     line_width=5,
                     show_scalar_bar=False)

    # Adiciona indicador de direcao
    plotter.add_axes(
        line_width=5,
        cone_radius=0.6,
        shaft_length=0.9,
        tip_length=0.2,
        ambient=0.5,
        label_size=(0.25, 0.15))

    # Adiciona limites do grafico
    plotter.show_bounds(
        grid='back',
        location='outer',
        ticks='both',
        n_xlabels=2,
        n_ylabels=2,
        n_zlabels=2,
        xtitle='x',
        ytitle='y',
        ztitle='z')

    # Cria grafico
    plotter.screenshot(filename + ".png")  # Save as screenshot
    plotter.show()


def plot_3d_planes(array_3d, x_offset=0., y_offset=0., z_offset=0., file_name="3D_planes"):
    # Definir as dimensões do array
    x_dim, y_dim, z_dim = array_3d.shape

    # Calcular o índice central para cada eixo
    x_center = x_dim // 2
    y_center = y_dim // 2
    z_center = z_dim // 2

    # Extrair os planos ao longo dos eixos centrais
    plane_x = array_3d[x_center, :, :]  # Plano no centro de x
    plane_y = array_3d[:, y_center, :]  # Plano no centro de y
    plane_z = array_3d[:, :, z_center]  # Plano no centro de z

    # Coordenadas para os planos com offset
    y_coords, z_coords = np.meshgrid(
        np.arange(y_dim) + y_offset, np.arange(z_dim) + z_offset, indexing="ij")
    x_coords, z_coords_x = np.meshgrid(
        np.arange(x_dim) + x_offset, np.arange(z_dim) + z_offset, indexing="ij")
    x_coords_y, y_coords_y = np.meshgrid(
        np.arange(x_dim) + x_offset, np.arange(y_dim) + y_offset, indexing="ij")

    # Calcular os limites da escala de cores
    # Valor mínimo em todo o array
    cmin = np.nanmin(np.concatenate((plane_x, plane_y, plane_z)))
    # Valor máximo em todo o array
    cmax = np.nanmax(np.concatenate((plane_x, plane_y, plane_z)))

    # Criar a figura com os três planos
    fig = go.Figure()

    # Adicionar plano em x
    fig.add_trace(go.Surface(
        x=np.full_like(plane_x, x_center + x_offset),
        y=y_coords,
        z=z_coords,
        surfacecolor=plane_x,
        colorscale='agsunset',
        cmin=cmin,  # Valor mínimo da escala de cores
        cmax=cmax,  # Valor máximo da escala de cores
        colorbar=dict(title="Valor (Plano X)"),
        lighting=dict(ambient=1, diffuse=0, specular=0, roughness=1),
    ))

    # Adicionar plano em y
    fig.add_trace(go.Surface(
        x=x_coords,
        y=np.full_like(plane_y, y_center + y_offset),
        z=z_coords_x,
        surfacecolor=plane_y,
        colorscale='agsunset',
        cmin=cmin,  # Valor mínimo da escala de cores
        cmax=cmax,  # Valor máximo da escala de cores
        colorbar=dict(title="Valor (Plano Y)"),
        lighting=dict(ambient=1, diffuse=0, specular=0, roughness=1),
    ))

    # Adicionar plano em z
    fig.add_trace(go.Surface(
        x=x_coords_y,
        y=y_coords_y,
        z=np.full_like(plane_z, z_center + z_offset),
        surfacecolor=plane_z,
        colorscale='agsunset',
        cmin=cmin,  # Valor mínimo da escala de cores
        cmax=cmax,  # Valor máximo da escala de cores
        colorbar=dict(title="Valor (Plano Z)"),
        lighting=dict(ambient=1, diffuse=0, specular=0, roughness=1),
    ))

    # Ajustar os limites do gráfico e rótulos
    fig.update_layout(
        title="Cortes 3D Interativos do Domínio com Offset",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            xaxis=dict(nticks=10, range=[x_offset, x_dim + x_offset]),
            yaxis=dict(nticks=10, range=[y_offset, y_dim + y_offset]),
            zaxis=dict(nticks=10, range=[z_offset, z_dim + z_offset])
        ),
    )

    # Salvar o gráfico em um arquivo HTML para visualização interativa
    fig.write_html(file_name)
    print(f"Gráfico 3D salvo como '{file_name}'")
    
def plot_heatmap(array_2d, output_file="heatmap_hd.png", dpi=300, cmap="inferno", xlabel="X-axis", ylabel="Y-axis", title="Heatmap", colorbar_label="Value", vmin=1, vmax=180):
    """
    Create and save a high-resolution heatmap from a 2D NumPy array with solid cell colors, gridlines, and adjustable color range.

    Parameters:
        array_2d (np.ndarray): 2D NumPy array to plot.
        output_file (str): File name to save the heatmap (e.g., "heatmap_hd.png").
        dpi (int): Resolution of the output image in dots per inch (default: 300).
        cmap (str): Matplotlib colormap to use (default: "inferno").
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the heatmap.
        colorbar_label (str): Label for the colorbar.
        vmin (float): Minimum value for the color scale (default: None, automatic scaling).
        vmax (float): Maximum value for the color scale (default: None, automatic scaling).
    """
    
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(16, 9))  # Full HD aspect ratio

    # Plot the heatmap with no interpolation (solid colors) and custom color range
    heatmap = ax.imshow(array_2d, cmap=cmap, aspect="auto",
                        origin="upper", interpolation="none", vmin=vmin, vmax=vmax)

    # Add gridlines for the grid view
    ax.set_xticks(np.arange(-0.5, array_2d.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, array_2d.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Add a colorbar
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label(colorbar_label)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Save the plot in HD resolution
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Heatmap saved as '{output_file}'")
    """
    plt.imshow(array_2d, cmap='viridis', interpolation=None)
    plt.colorbar()
    plt.show()
