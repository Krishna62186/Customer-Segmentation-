# Customer Segmentation with K-Means

Unsupervised customer segmentation on a demographic dataset of **2,627 customers**. The notebook cleans the data, picks a cluster count, fits K-Means, then profiles and stress-tests the resulting segments.

## Dataset

`Test.csv` has 2,627 rows and 10 columns.

| Column | Type | Description |
|---|---|---|
| `ID` | int | Customer ID (dropped before clustering) |
| `Gender` | categorical | Female / Male |
| `Ever_Married` | categorical | Yes / No |
| `Age` | numeric | Age in years |
| `Graduated` | categorical | Yes / No |
| `Profession` | categorical | Artist, Doctor, Engineer, Entertainment, Executive, Healthcare, Homemaker, Lawyer, Marketing |
| `Work_Experience` | numeric | Years of work experience |
| `Spending_Score` | ordinal | Low / Average / High |
| `Family_Size` | numeric | Number of family members |
| `Var_1` | categorical | Anonymised category (`Cat_1` to `Cat_7`) |

The file is not included in this repo. Place `Test.csv` next to the notebook before running.

## Approach

1. **Exploratory analysis**: missing values, distributions, categorical counts, correlation heatmap, and IQR-based outlier counts (outliers were flagged but kept).
2. **Preprocessing**
   - Missing numeric values filled with the median, categorical with the mode.
   - `Spending_Score` mapped as an ordinal (Low=0, Average=1, High=2).
   - Remaining categoricals one-hot encoded (`drop_first=True`), giving 21 features.
   - Features standardised with `StandardScaler`.
3. **Choosing k**: elbow method and silhouette score for k = 2 to 10, plus Davies-Bouldin and Calinski-Harabasz. **k = 4** was selected.
4. **Clustering**: `KMeans(n_clusters=4, random_state=42, n_init=10)`.
5. **Visualisation**: PCA and t-SNE scatter plots, a standardised cluster-means heatmap, and a radar chart of cluster profiles.
6. **Validation**: silhouette plot per cluster, plus stability checks using Adjusted Rand Index across random seeds and 80% subsamples.

## Results

| Cluster | Size | Avg. age | Avg. work exp. | Avg. family size | Top profession | Top spending | Married | Graduated | Auto-name |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1,142 | 54.8 | 2.1 | 2.6 | Artist | Average | 97% | 76% | Middle-aged Average-spenders |
| 1 | 834 | 31.9 | 3.3 | 2.8 | Healthcare | Low | 8% | 54% | Young Low-spenders |
| 2 | 384 | 39.0 | 2.2 | 3.6 | Engineer | Low | 54% | 39% | Middle-aged Low-spenders |
| 3 | 267 | 39.6 | 2.6 | 2.9 | Artist | Low | 51% | 53% | Middle-aged Low-spenders |

**What separates the segments**
- **Cluster 0**: the oldest group, almost entirely married, most likely to be graduates, and the only group whose most common spending level is Average.
- **Cluster 1**: the youngest group, overwhelmingly unmarried, with the most work experience and a large Healthcare share.
- **Cluster 2**: the largest families and the lowest graduate rate, with a high share of Engineers.
- **Cluster 3**: sits between the others demographically and overlaps with Cluster 2 on spending behaviour.

### Cluster quality (k = 4)

| Metric | Value |
|---|---|
| Silhouette | 0.145 |
| Davies-Bouldin (lower is better) | 2.311 |
| Calinski-Harabasz (higher is better) | 233.9 |
| Points with negative silhouette | Cluster 1: 10.1%, all others: 0% |
| Stability (ARI), different seeds | mean 0.441, min 0.106 |
| Stability (ARI), 80% subsamples | mean 0.315, min 0.154 |

### Limitations

The cluster structure in this data is **weak**:

- A silhouette of 0.145 indicates heavily overlapping clusters, and the ARI stability scores fall well below the ~0.8 level usually considered good.
- Clusters 2 and 3 end up with the same auto-generated persona name, so they are only loosely distinguishable.
- Silhouette and Davies-Bouldin improve as k grows toward 10 in this run, so k = 4 is a judgment call balancing interpretability against fit, not a clear optimum.
- The data is mostly categorical, which K-Means (Euclidean distance on one-hot features) handles imperfectly. K-Prototypes, K-Modes, or Gower distance with hierarchical clustering are worth trying.
- Cluster labels can shift between runs and data samples, so treat the personas as exploratory.

## Getting started

### Run in Google Colab
Open the notebook in Colab and run all cells. The first cell prompts you to upload `Test.csv`.

### Run locally
```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook customer_segmentation.ipynb
```

The first cell uses `google.colab.files.upload()`, which only works in Colab. Locally, remove that import and the upload line, and read the file directly:

```python
df = pd.read_csv("Test.csv")
```

## Tech stack

Python, pandas, NumPy, scikit-learn (KMeans, PCA, t-SNE, clustering metrics), Matplotlib, Seaborn.

## Repository structure

```
.
├── customer_segmentation.ipynb   # Full analysis
├── Test.csv                      # Dataset (add this yourself)
└── README.md
```

## Possible next steps

- Try K-Prototypes or Gower distance with hierarchical clustering for mixed data types.
- Compare against Gaussian Mixture Models or DBSCAN.
- Run a feature-importance or feature-selection pass to see which variables actually drive separation.
- Name the personas manually and attach marketing actions (the notebook has a `personas` dictionary ready for this).
