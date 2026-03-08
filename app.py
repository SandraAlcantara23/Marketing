from flask import Flask, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
 
app = Flask(__name__)

# Cargar dataset
df = pd.read_csv("sales_data_sample.csv", encoding="latin1")

# Información del dataset
num_rows = df.shape[0]
num_cols = df.shape[1]
columns = df.columns.tolist()

# valores nulos
missing = df.isnull().sum().to_dict()

# estadísticas
stats = df.describe().to_html(classes="table table-striped")

# Top países por ventas
sales_country = df.groupby("COUNTRY")["SALES"].sum().sort_values(ascending=False).head(10)
sales_country = sales_country.to_frame().to_html(classes="table table-striped")

# Datos para clustering
data = df[['SALES','QUANTITYORDERED','PRICEEACH']]

scaler = StandardScaler()
scaled = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=4, n_init=10)
clusters = kmeans.fit_predict(scaled)

pca = PCA(n_components=3)
pca_data = pca.fit_transform(scaled)

pca_df = pd.DataFrame(pca_data, columns=['pca1','pca2','pca3'])
pca_df["cluster"] = clusters


@app.route("/")
def dashboard():

    # Cluster 3D
    fig1 = px.scatter_3d(
        pca_df,
        x='pca1',
        y='pca2',
        z='pca3',
        color='cluster',
        title="Clusters PCA 3D"
    )

    # Histograma
    fig2 = px.histogram(
        df,
        x="SALES",
        title="Distribución de Ventas"
    )

    # Scatter Matrix
    fig3 = px.scatter_matrix(
        data,
        title="Relación entre variables"
    )

    # Correlación
    corr = df.corr(numeric_only=True)

    fig4 = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="magma",
        title="Matriz de Correlación"
)

    fig4.update_layout(
        width=900,
        height=700
)

    return render_template(
        "dashboard.html",
        graph1=fig1.to_html(full_html=False),
        graph2=fig2.to_html(full_html=False),
        graph3=fig3.to_html(full_html=False),
        graph4=fig4.to_html(full_html=False),
        rows=num_rows,
        cols=num_cols,
        columns=columns,
        missing=missing,
        stats=stats,
        sales_country=sales_country
    )


if __name__ == "__main__":
    app.run(debug=True)