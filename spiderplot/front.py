import streamlit as st

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np

st.set_page_config(layout="wide")

st.markdown("""
## 🕸️ Sensory Spider Plot Generator

Upload your sensory evaluation CSV file and compare up to 5 different samples in a radar plot.
You can optionally add confidence bounds (e.g., Upper/Lower) to visualize expected ranges.
""")


st.markdown("### 📁 Upload CSV")

file = st.file_uploader("Upload csv file with sensory evaluation")

if file:
    plot_info = pd.read_csv(file, sep= ';')
    
df = plot_info.rename(columns={plot_info.columns[0]: "Attribute"})
df.set_index("Attribute", inplace=True)

st.dataframe(df)

st.markdown("---")

col1, col2 = st.columns(2)

with col2:
    font_size = st.number_input("Labels font size", min_value=6.0, step=0.5)

with col1:
    sample_count = st.number_input("Number of samples to plot", min_value=1, step=1, value=1)

sample_names = []
sample_colors = []

st.markdown("### Plot configuration")
default_colors = ['#00b8ff', '#307af3', '#9fce47', '#ffaf40', '#fc5185']

col3, col4 = st.columns(2)

with col3:
    with st.expander("⚙️ Samples selection and configuration"):
        if file:
            available_columns = [col for col in df.columns if col not in []]

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


def generate_plot(df, font, sample_names, sample_colors, lower_bound=None, upper_bound=None):
    labels = df.index.tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += [angles[0]]
    labels += [labels[0]]

    labels_upper = [label.upper() for label in labels[:-1]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Reemplazo del fondo circular por decágono
    num_vars = len(labels) - 1
    ax.set_frame_on(False)
    ax.patch.set_visible(False)
    decagon = polygon_patch(num_vars)
    ax.add_patch(decagon)
    decagon.set_facecolor('white')
    decagon.set_edgecolor('lightgray')
    decagon.set_alpha(1)
    decagon.set_zorder(0)

    # Dibujar áreas de bounds si existen
    if upper_bound and lower_bound:
        upper = df[upper_bound].tolist() + [df[upper_bound].iloc[0]]
        lower = df[lower_bound].tolist() + [df[lower_bound].iloc[0]]
        ax.fill(angles, upper, color='gray', alpha=0.2, label='Bounds')
        ax.fill(angles, lower, color='white', alpha=1)
    elif upper_bound:
        upper = df[upper_bound].tolist() + [df[upper_bound].iloc[0]]
        lower = [0] * (len(upper) - 1) + [0]
        ax.fill(angles, upper, color='gray', alpha=0.2, label='Upper Bound')
        ax.fill(angles, lower, color='white', alpha=1)
    elif lower_bound:
        lower = df[lower_bound].tolist() + [df[lower_bound].iloc[0]]
        max_val = df.max().max()
        upper = [max_val] * (len(lower) - 1) + [max_val]
        ax.fill(angles, upper, color='gray', alpha=0.2, label='Lower Bound')
        ax.fill(angles, lower, color='white', alpha=1)

    # Graficar muestras
    for sample, color in zip(sample_names, sample_colors):
        if sample in df.columns:
            values = df[sample].tolist() + [df[sample].iloc[0]]
            ax.plot(angles, values, color=color, linewidth=2, label=sample)
            ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids([])

    for angle, label in zip(angles[:-1], labels_upper):
        x = angle
        y = ax.get_rmax() * 1.2
        ax.text(x, y, label, ha='center', va='center', fontsize=font, fontweight='bold')

    ax.legend(loc='upper left', bbox_to_anchor=(-0.35, 1.1))
    fig.tight_layout()
    return fig
    
# Bloque de configuración de límites
with col4:
    with st.expander("⚙️ Optional bounds configuration"):
        bound_options = st.multiselect("Select which bounds exist in your data", options=['Upper Bound', 'Lower Bound'])

        upper_bound_col = None
        lower_bound_col = None

        if file:
            column_options = df.columns.tolist()

            if 'Upper Bound' in bound_options:
                upper_bound_col = st.selectbox("Select the column for Upper Bound", options=column_options, key="ub")
            if 'Lower Bound' in bound_options:
                lower_bound_col = st.selectbox("Select the column for Lower Bound", options=column_options, key="lb")

# Luego al llamar la función de graficado:
figure = generate_plot(df, font_size, sample_names, sample_colors, lower_bound_col, upper_bound_col)

st.pyplot(figure)
