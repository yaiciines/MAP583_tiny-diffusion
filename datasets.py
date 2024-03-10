import numpy as np
import pandas as pd
import torch

from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset
import tensorflow as tf
#==============================================================================================================================================================
# Explication of the code
#==============================================================================================================================================================
"""Le fichier `datasets.py` contient des fonctions pour générer différents ensembles de données synthétiques utilisables pour l'entraînement de modèles en machine 
learning. Voici un résumé des fonctions présentes :

1. **`moons_dataset(n=8000)`** : Génère un ensemble de données en forme de deux lunes imbriquées. `n` spécifie le nombre d'échantillons. Les données sont 
transformées et retournées sous forme de `TensorDataset` compatible avec PyTorch.

2. **`line_dataset(n=8000)`** : Crée un ensemble de données linéaire. Les points sont générés uniformément dans un espace défini et multipliés par un facteur 
pour étendre leur distribution. Le résultat est également retourné comme un `TensorDataset`.

3. **`circle_dataset(n=8000)`** : Produit un ensemble de données circulaire. Les points sont d'abord générés et normalisés pour former un cercle, puis légèrement 
déplacés pour ajouter du bruit. Comme les autres, il est retourné sous forme de `TensorDataset`.

4. **`dino_dataset(n=8000)`** : Charge et traite un ensemble de données spécifique nommé "dino" à partir d'un fichier TSV. Les points sont ajustés avec un bruit 
normal et redimensionnés avant d'être retournés comme `TensorDataset`.

5. **`get_dataset(name, n=8000)`** : Une fonction utilitaire pour obtenir l'un des ensembles de données ci-dessus en fonction du nom fourni. Lance une exception 
si le nom n'est pas reconnu.

Chaque fonction utilise `numpy` pour la manipulation des données et `torch` pour convertir les données en format compatible avec PyTorch, facilitant leur 
utilisation dans des tâches d'apprentissage automatique."""

#==============================================================================================================================================================
# Génération de nouveaux ensembles de données
#==============================================================================================================================================================
def blobs_dataset(n=10000, centers=3, cluster_std=1.0):
    from sklearn.datasets import make_blobs
    import torch
    from torch.utils.data import TensorDataset
    
    X, y = make_blobs(n_samples=n, centers=centers, cluster_std=cluster_std, random_state=42)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)

def concentric_circles_dataset(n=8000, noise=0.05):
    from sklearn.datasets import make_circles
    import torch
    from torch.utils.data import TensorDataset
    
    X, y = make_circles(n_samples=n, noise=noise, factor=0.5, random_state=42)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)

def s_curve_dataset(n=8000, noise=0.05):
    from sklearn.datasets import make_s_curve
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset
    
    X, y = make_s_curve(n_samples=n, noise=noise, random_state=42)
    X = X[:, [0, 2]]  # sélection des colonnes pour obtenir un effet 2D
    y = (y > np.median(y)).astype(int)  # binarisation des étiquettes
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)

# différentes lignes : 
def horizontal_lines_dataset(n=8000, lines=1):
    x = np.linspace(-1, 1, n // lines)
    y_positions = np.linspace(-1, 1, lines)
    data = np.array([(xi, y + np.random.uniform(-0.5, 0.5)) for y in y_positions for xi in x])
    return TensorDataset(torch.tensor(data, dtype=torch.float32))

def vertical_lines_dataset(n=8000, lines=1):
    y = np.linspace(-1, 1, n // lines)
    x_positions = np.linspace(-1, 1, lines)
    data = np.array([(x + np.random.uniform(-0.5, 0.5), yi) for x in x_positions for yi in y])
    return TensorDataset(torch.tensor(data, dtype=torch.float32))

def inclined_lines_dataset(n=8000, angle_degrees=45, lines=1):
    angle = np.radians(angle_degrees)
    x = np.linspace(-1, 1, n // lines)
    data = np.array([(xi, np.tan(angle) * xi + np.random.uniform(-0.5, 0.5)) for xi in x for _ in range(lines)])
    return TensorDataset(torch.tensor(data, dtype=torch.float32))

#==============================================================================================================================================================


def adjust_poisson_values(poisson_val, max_val=0.5):
    # Transformation des valeurs Poisson pour une concentration sur les bords.
    return np.where(poisson_val < max_val, poisson_val, max_val * 2 - poisson_val)

def horizontal_lines_dataset_poisson(n=8000, lines=1, lam=1.0):
    x = np.linspace(-1, 1, n // lines)
    y_positions = np.linspace(-1, 1, lines)
    data = np.array([(xi, y + adjust_poisson_values(np.random.poisson(lam) / 10.0 - 0.5)) for y in y_positions for xi in x])
    return TensorDataset(torch.tensor(data, dtype=torch.float32))

def vertical_lines_dataset_poisson(n=8000, lines=1, lam=1.0):
    y = np.linspace(-1, 1, n // lines)
    x_positions = np.linspace(-1, 1, lines)
    data = np.array([(x + adjust_poisson_values(np.random.poisson(lam) / 10.0 - 0.5), yi) for x in x_positions for yi in y])
    return TensorDataset(torch.tensor(data, dtype=torch.float32))

def inclined_lines_dataset_poisson(n=8000, angle_degrees=45, lines=1, lam=1.0):
    angle = np.radians(angle_degrees)
    x = np.linspace(-1, 1, n // lines)
    data = np.array([(xi, np.tan(angle) * xi + adjust_poisson_values(np.random.poisson(lam) / 10.0 - 0.5)) for xi in x for _ in range(lines)])
    return TensorDataset(torch.tensor(data, dtype=torch.float32))

#==============================================================================================================================================================
# MNIST 
#==============================================================================================================================================================

def load_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    i = np.random.randint(60000) 
    return train_images[i]
# Fonction pour ajouter un carré de points autour de chaque point existant
def add_square_interpolation(x, y, density):
    new_x, new_y = [], []
    
    for xi, yi in zip(x, y):
      new_x = np.concatenate([new_x, np.linspace(xi - 1, xi + 1, density)])
      new_y = np.concatenate([new_y, np.linspace(yi - 1, yi + 1, density)])

    return new_x, new_y

def mnist_dataset():
    train_image = load_mnist_data()
    density = 100
    x=[]
    y=[]
    for i in range(28):
        for j in range(28):
            pixel = train_image[i][j]
            if pixel > 0:
                x.append(j)
                y.append(28 - i)
                
    new_x, new_y = add_square_interpolation(x, y, density)
    X = np.stack((new_x, new_y), axis=1)

    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

# ==============================================================================================================================================================

def moons_dataset(n=8000):
    X, _ = make_moons(n_samples=n, random_state=42, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def line_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X *= 4
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def circle_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    y = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    norm = np.sqrt(x**2 + y**2) + 1e-10
    x /= norm
    y /= norm
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    r = rng.uniform(0, 0.03, n)
    x += r * np.cos(theta)
    y += r * np.sin(theta)
    X = np.stack((x, y), axis=1)
    X *= 3
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def dino_dataset(n=8000):
    df = pd.read_csv("static/DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    x = (x/54 - 1) * 4
    y = (y/48 - 1) * 4
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def random_dataset(n=8000):
    # return a random dataset from the list
    datasets = ["moons", "circle", "dino"]
    return get_dataset(np.random.choice(datasets), n)



def get_dataset(name, n=8000):
    # changed 
    if name == "blobs":
        return blobs_dataset(n)
    elif name == "concentric_circles":
        return concentric_circles_dataset(n)
    elif name == "s_curve":
        return s_curve_dataset(n)
    
    elif name == "horizontal_lines":
        return horizontal_lines_dataset(n)
    elif name == "vertical_lines":
        return vertical_lines_dataset(n)
    elif name == "inclined_lines":
        return inclined_lines_dataset(n)
    
    elif name == "horizontal_lines_poisson":
        return horizontal_lines_dataset_poisson(n)
    elif name == "vertical_lines_poisson":
        return vertical_lines_dataset_poisson(n)
    elif name == "inclined_lines_poisson":
        return inclined_lines_dataset_poisson(n)
    
    elif name == "mnist":
        return mnist_dataset()
    
    elif name == "random":
        return random_dataset(n)
     
    if name == "moons":
        return moons_dataset(n)
    elif name == "dino":
        return dino_dataset(n)
    elif name == "line":
        return line_dataset(n)
    elif name == "circle":
        return circle_dataset(n)
    else:
        raise ValueError(f"Unknown dataset: {name}")
