import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np
import csv
import io
from pathlib import Path

# Configuración general
st.set_page_config(layout="wide")


# Ruta absoluta del directorio que contiene este script (front.py)
BASE_DIR = Path(__file__).resolve().parent

# Ruta al archivo 'logo_black.png'
logo_black = BASE_DIR / "logo_black.png"

# Estilos globales
HEADER_COLOR = "#ff0093"
st.markdown(f"""<h2 style='color:{HEADER_COLOR}; font-weight: bold;'>🕸️ Sensory Spider Plot Generator</h2>""", unsafe_allow_html=True)
st.markdown("""
Upload your sensory evaluation CSV file and compare up to 5 different samples in a radar plot.
You can optionally add confidence bounds (e.g., Upper/Lower) to visualize expected ranges.
""")

# Placeholder para logo (puedes reemplazar por `st.image("path/to/logo.png", width=150)`)
st.markdown("---")
with open(logo_black, "rb") as f:
    st.image(f.read())
st.markdown("---")

# Subheader
st.markdown(f"""<h4 style='color:{HEADER_COLOR}; font-weight: bold;'>📁 Upload CSV</h4>""", unsafe_allow_html=True)

col_file, col_sep = st.columns([3, 1])

with col_file:
    file = st.file_uploader("Upload csv file with sensory evaluation")

if file:

    # Intento de autodetección
    content = file.read().decode("utf-8")
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(content[:1024])
    default_sep = dialect.delimiter

    # Permitir corrección manual si la inferencia no fue adecuada
    with col_sep:
        sep = st.selectbox("Detected separator (you can change it if needed):", options=[",", ";", "\t", "|"], index=[",", ";", "\t", "|"].index(default_sep))
    file.seek(0)
    plot_info = pd.read_csv(file, sep=sep)
    
    df = plot_info.rename(columns={plot_info.columns[0]: "Attribute"})
    df.set_index("Attribute", inplace=True)

    st.markdown(f"""<h4 style='color:{HEADER_COLOR}; font-weight: bold;'>📊 Preview of Uploaded Data</h4>""", unsafe_allow_html=True)
    
    # Redondear solo columnas numéricas
    df_display = df.copy()
    for col in df_display.select_dtypes(include=[np.number]).columns:
        df_display[col] = df_display[col].map(lambda x: round(x, 2))

    num_columns = len(df_display.columns)
    column_width = f"{60 / num_columns:.1f}%"

    styled = df_display.style\
        .set_properties(**{
            "text-align": "center",
            "font-size": "14px",
            "font-weight": "bold",
        })\
        .set_table_styles([
            # Estilo para headers (columnas)
            {"selector": "th.col_heading", "props": [
                ("background-color", "#ff0091"),
                ("color", "white"),
                ("font-weight", "bold"),
                ("text-align", "center !important"),
                ("width", column_width),
            ]},
            # Estilo para índice (columna izquierda)
            {"selector": "th.row_heading", "props": [
                ("background-color", "black"),
                ("color", "white"),
                ("font-weight", "bold"),
                ("width", "15%"),  # o el valor que te acomode para la columna de atributos
            ]},
            # Estilo para celdas normales
            {"selector": "td", "props": [
                ("text-align", "center !important"),
                ("width", column_width),
            ]}
        ])

    st.table(styled)

    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col2:
        font_size = st.number_input("Labels font size", min_value=5.0, step=0.5, value=6.0)

    with col1:
        sample_count = st.number_input("Number of samples to plot", min_value=1, step=1, value=1)

    sample_names = []
    sample_colors = []
    default_colors = ['#00b8ff', '#307af3', '#9fce47', '#ffaf40', '#fc5185']

    st.markdown(f"""<h4 style='color:{HEADER_COLOR}; font-weight: bold;'>🧪 Plot Configuration</h4>""", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        with st.expander("⚙️ Samples selection and configuration"):
            available_columns = df.columns.tolist()
            for i in range(sample_count):
                name = st.selectbox(f"Sample {i+1} column", options=available_columns, key=f"sample_{i}")
                color = st.color_picker(f"Sample {i+1} color", value=default_colors[i % len(default_colors)])
                sample_names.append(name)
                sample_colors.append(color)

    def polygon_patch(num_vars, radius=1.0):
        theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
        verts = [(np.cos(t) * radius, np.sin(t) * radius) for t in theta]
        verts.append(verts[0])
        return mpatches.PathPatch(mpath.Path(verts), transform=None)

    with col4:
        with st.expander("⚙️ Optional bounds configuration"):
            bound_options = st.multiselect("Select which bounds exist in your data", options=['Upper Bound', 'Lower Bound'])

            upper_bound_col = None
            lower_bound_col = None

            column_options = df.columns.tolist()
            if 'Upper Bound' in bound_options:
                upper_bound_col = st.selectbox("Select the column for Upper Bound", options=column_options, key="ub")
            if 'Lower Bound' in bound_options:
                lower_bound_col = st.selectbox("Select the column for Lower Bound", options=column_options, key="lb")

    # Selector de posición de la leyenda
    legend_pos = st.selectbox("📍 Legend position", options=[
        ("Top right", ("upper right", (1.3, 1.1))),
        ("Top left", ("upper left", (-0.4, 1.1))),
    ], format_func=lambda x: x[0])

    def generate_plot(df, font, sample_names, sample_colors, lower_bound=None, upper_bound=None):
        labels = df.index.tolist()
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += [angles[0]]
        labels += [labels[0]]
        labels_upper = [label.upper() for label in labels[:-1]]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        
        ax.tick_params(labelsize=font * 0.9)

        num_vars = len(labels) - 1
        ax.set_frame_on(False)
        ax.patch.set_visible(False)
        decagon = polygon_patch(num_vars)
        ax.add_patch(decagon)
        decagon.set_facecolor('white')
        decagon.set_edgecolor('lightgray')
        decagon.set_alpha(1)
        decagon.set_zorder(0)

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
            ax.text(x, y, label, ha='center', va='center', fontsize=font*0.8, fontweight='bold')

        legend_loc, legend_anchor = legend_pos[1]  # unpack
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_anchor, fontsize=font * 1.05)
        fig.tight_layout()
        return fig

    st.markdown("---")
    figure = generate_plot(df, font_size, sample_names, sample_colors, lower_bound_col, upper_bound_col)
    col_l, col_plot, col_r = st.columns([1, 2, 1])
    with col_plot:
        st.pyplot(figure)
