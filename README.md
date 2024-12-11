![Project](https://img.shields.io/badge/Project-rdga_4k-blue)
![Author](https://img.shields.io/badge/Author-aquinordg-green)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Version](https://img.shields.io/badge/Version-1.0-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

# rdga_4k (Random Data Generator Algorithm for Clustering)

The rdga_4k library generates synthetic datasets tailored for clustering algorithm applications. It provides two core functions, `catbird` and `canard`, for customizable dataset generation with support for binary and categorical features.

---

## 🔥 Features

- **Synthetic Data for Clustering:** Tailored datasets for clustering algorithm research and testing.
- **Flexible Configurations:** Supports binary and categorical feature generation.
- **Noise and Intersection Control:** Fine-tune feature noise and cluster intersections.
- **Reproducible Results:** Ensure consistency with random seed support.

---

## 🛠 Installation

Install using *git* and *pip install*:

```bash
pip install git+https://github.com/aquinordg/rdga_4k.git

```

---

## 🚀 Usage

Import the library and use the `catbird` or `canard` functions to generate datasets:

```python
from rdga_4k import catbird, canard

# Example using catbird
X, y = catbird(
    n_feat=10,
    feat_sig=[3, 2],
    rate=[50, 50],
    lmbd=0.7,
    eps=0.1,
    random_state=42
)

# Example using canard
X, y = canard(
    n_feat=10,
    n_cat=3,
    rate=[50, 50],
    lmbd=5,
    eps=0.2,
    random_state=42
)
```

---

## 📜 Functions Overview

### `catbird`

Generates a labeled dataset with binary features based on feature clustering.

#### Parameters

- `n_feat` (int): Number of total features. Must be greater than 1.
- `feat_sig` (list[int]): List of the number of significant features per cluster.
- `rate` (list[int]): Number of examples per cluster.
- `lmbd` (float): Intersection factor between features. Default is `0.8`.
- `eps` (float): Noise rate for feature generation. Default is `0.2`.
- `random_state` (int or RandomState, optional): Seed for reproducibility.

#### Returns
- `X` (np.ndarray): Binary matrix representing the features.
- `y` (np.ndarray): Array of cluster labels.

#### Example

```python
X, y = catbird(n_feat=10, feat_sig=[3, 2], rate=[50, 50], lmbd=0.7, eps=0.1, random_state=42)
```

---

### `canard`

Generates a labeled dataset with categorical features divided into multiple categories.

#### Parameters

- `n_feat` (int): Number of total features. Must be greater than 1.
- `n_cat` (int): Number of categories for each feature. Must be greater than 1.
- `rate` (list[int]): Number of examples per cluster.
- `lmbd` (int): Intersection factor between features. Default is `10`.
- `eps` (float): Noise rate for feature generation. Default is `0.3`.
- `random_state` (int or RandomState, optional): Seed for reproducibility.

#### Returns
- `X` (np.ndarray): Matrix of categorical features.
- `y` (np.ndarray): Array of cluster labels.

#### Example

```python
X, y = canard(n_feat=10, n_cat=3, rate=[50, 50], lmbd=5, eps=0.2, random_state=42)
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Push to the branch.
5. Open a pull request.

For questions or information, feel free to reach out at: [aquinordga@gmail.com](mailto:aquinordga@gmail.com).

---

## 👨‍💻 Author

AQUINO, R. D. G.
[![Lattes](https://github.com/aquinordg/custom_tools/blob/main/icons/icons8-plataforma-lattes-100.png)](http://lattes.cnpq.br/2373005809061037)
[![ORCID](https://github.com/aquinordg/custom_tools/blob/main/icons/icons8-orcid-100.png)](https://orcid.org/0000-0002-8486-8354)
[![Google Scholar](https://github.com/aquinordg/custom_tools/blob/main/icons/icons8-google-scholar-100.png)](https://scholar.google.com/citations?user=r5WsvKgAAAAJ&hl)

---

## 💬 Feedback

Feel free to open an issue or contact me for feedback or feature requests. Your input is highly appreciated!
