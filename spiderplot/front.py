import streamlit as st

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np

st.header("Spider Plot Generator")

##Default values for testing
upper_bound = 'Upper Bound'
lower_bound = 'Lower Bound'
##End


file = st.file_uploader("Upload csv file with sensory evaluation")

if file:
    plot_info = pd.read_csv(file, sep= ';')
    
df = plot_info.rename(columns={plot_info.columns[0]: "Attribute"})
df.set_index("Attribute", inplace=True)

st.dataframe(df)

font_size = st.number_input("labels font size", min_value = 6.0, step = 0.5)

sample_count = st.number_input("Number of samples to plot", min_value=1, step=1, value=1)

sample_names = []
sample_colors = []

st.markdown("### Sample configuration")
default_colors = ['#00b8ff', '#307af3', '#9fce47', '#ffaf40', '#fc5185']

if file:
    available_columns = [col for col in df.columns if col not in [upper_bound, lower_bound]]

    for i in range(sample_count):
        name = st.selectbox(f"Sample {i+1} column", options=available_columns, key=f"sample_{i}")
        color = st.color_picker(f"Sample {i+1} color", value=default_colors[i % len(default_colors)])
        sample_names.append(name)
        sample_colors.append(color)

def polygon_patch(num_vars, radius=1.0):
    """Crea un Path de polígono regular para usar como fondo del radar chart."""
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    verts = [(np.cos(t) * radius, np.sin(t) * radius) for t in theta]
    verts.append(verts[0])  # Cerrar el polígono
    return mpatches.PathPatch(mpath.Path(verts), transform=None)


def generate_plot(df, font, sample_names, sample_colors, lower_bound='Lower Bound', upper_bound='Upper Bound'):
    
    # Etiquetas de atributos
    labels = df.index.tolist()

    # Crear ángulos para cada eje (10 atributos + 1 para cerrar)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += [angles[0]]  # Cierra el círculo

    # Valores de cada muestra + repetir primer valor
    upper = df[upper_bound].tolist() + [df[upper_bound].iloc[0]]
    lower = df[lower_bound].tolist() + [df[lower_bound].iloc[0]]
    labels += [labels[0]]  # Para alinear con thetagrids

    # Convertir etiquetas a mayúsculas
    labels_upper = [label.upper() for label in labels[:-1]]
    
    # Iniciar figura
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Estilo: desactivar fondo circular
    ax.set_frame_on(False)

    # Reemplazar el fondo circular por un decágono
    num_vars = len(labels) - 1  # 10 atributos
    ax.patch.set_visible(False)
    decagon = polygon_patch(num_vars)
    ax.add_patch(decagon)
    decagon.set_facecolor('white')
    decagon.set_edgecolor('lightgray')
    decagon.set_alpha(1)
    decagon.set_zorder(0)

    # Dibujar área entre bounds
    ax.fill(angles, upper, color='gray', alpha=0.2, label='Bounds')
    ax.fill(angles, lower, color='white', alpha=1)

    # Dibujar muestras con colores personalizados
    for sample, color in zip(sample_names, sample_colors):
        if sample in df.columns:
            values = df[sample].tolist() + [df[sample].iloc[0]]
            ax.plot(angles, values, color=color, linewidth=2, label=sample)
            ax.fill(angles, values, color=color, alpha=0.1)

    # Configuración del eje y etiquetas
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids([])

    # Agregar etiquetas en mayúscula, negrita, más lejos del centro
    labels_upper = [label.upper() for label in labels[:-1]]
    for angle, label in zip(angles[:-1], labels_upper):
        x = angle
        y = ax.get_rmax() * 1.2
        ax.text(
            x, y, label,
            ha='center',
            va='center',
            fontsize=font,
            fontweight='bold',
            #rotation=np.degrees(x),
            #rotation_mode='anchor'
        )

    # Leyenda en esquina superior izquierda
    ax.legend(loc='upper left', bbox_to_anchor=(-0.35, 1.1))

    fig.tight_layout()
    return fig
    
figure = generate_plot(df, font_size, sample_names, sample_colors)

st.pyplot(figure)
