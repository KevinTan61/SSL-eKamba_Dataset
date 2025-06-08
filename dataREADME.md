# Datasets used in SSL-eKamba

---

## 📊 Dataset Description
We provide several datasets used in the SSL-eKamba framework, which employs self-supervised learning for traffic accidents prediction. Collected from two public real-world sourcesn **[NYC](https://opendata.cityofnewyork.us/)** and **[Chicago](https://data.cityofchicago.org/browse?q=traffic&sortBy=relevance&pageSize=20)** these datasets are designed for spatiotemporal analysis and risk prediction tasks, with a detailed description including their structure, features, and associated metadata provided below.
### Overview

The datasets cover the period from **January 1 to May 31, 2023**, with a temporal resolution of **1 hour** (total of 3624 time steps for the 5-month period). They include traffic accident data, taxi order data, POI (Point of Interest) data, weather data, and road segment data. These datasets are processed into grid-based and graph-based formats for use in machine learning models, such as spatiotemporal graph neural networks.
#### Dataset Statistics
The key statistics of the NYC and Chicago datasets are summarized in the table below:

**Table 1: Dataset Statistics**

| Dataset   | Time Range    | Grid Size (W×H) | Nodes (N) | Risk Graph | Road Graph | POI Graph |
|-----------|---------------|-----------------|-----------|-----------|------------|-----------|
| NYC       | 2023.1–2023.5 | 20*20           | 243       | ✅         | ✅         | ✅        |
| Chicago   | 2023.6–2023.9 | 20*20           | 197       | ✅         | ✅         | ❌        |


---

## 📂 File Structure and Details

The datasets are structured within the `datasets` directory, with separate subdirectories for **NYC** and **Chicago**, each containing relevant data files tailored to their respective characteristics. Below is a concise summary of the files and their purposes:

- **Directory Structure**:
  ```
  datasets/
  ├── NYC/
  │   ├── spatiotemporal_features.pkl
  │   ├── grid_to_node_mapping.pkl
  │   ├── poi_similarity_adjacency.pkl
  │   ├── risk_similarity_adjacency.pkl
  │   ├── high_risk_mask.pkl
  │   └── road_network_adjacency.pkl
  ├── CHICAGO/
  │   ├── spatiotemporal_features.pkl
  │   ├── grid_to_node_mapping.pkl
  │   ├── risk_similarity_adjacency.pkl
  │   ├── high_risk_mask.pkl
  │   └── road_network_adjacency.pkl
  ```

###  File Details

1. **spatiotemporal_features.pkl**
   - **Purpose**: Core dataset with spatiotemporal features for 2023 (hourly data, 3624 time steps).
   - **Shape**: `(T, D, W, H)` where `T=3624`, `D=48`, `W` and `H` are grid dimensions (to be specified).
   - **Features**:
     - `0`: `risk` - Risk score (e.g., from accident data).
     - `1–24`: `time_period` - One-hot hour encoding (24 dims).
     - `25–31`: `day_of_week` - One-hot day encoding (7 dims).
     - `32`: `holiday` - One-hot holiday indicator.
     - `33–39`: `POI` - POI categories (7 dims, NYC only: residence, school, etc.).
     - `40`: `temperature` - Temperature value.
     - `41–45`: Weather (one-hot: `Clear`, `Cloudy`, `Rain`, `Snow`, `Mist`).
     - `46`: `inflow` - Traffic inflow.
     - `47`: `outflow` - Traffic outflow.

2. **high_risk_mask.pkl**
   - **Purpose**: Binary mask for high-risk grid regions.
   - **Shape**: `(W, H)` (1 = high-risk, 0 = otherwise).
   - **Use**: Focuses analysis on critical areas.

3. **risk_similarity_adjacency.pkl**
   - **Purpose**: Adjacency matrix for risk similarity graph.
   - **Shape**: `(N, N)` 
   - **Details**: Based on traffic accident similarity.
   - **Availability**: Both NYC and Chicago.

4. **road_network_adjacency.pkl**
   - **Purpose**: Adjacency matrix for road similarity graph.
   - **Shape**: `(N, N)`.
   - **Details**: Derived from road segment data (length, width, type, etc.).
   - **Availability**: Both NYC and Chicago.

5. **poi_similarity_adjacency.pkl**
   - **Purpose**: Adjacency matrix for POI similarity graph.
   - **Shape**: `(N, N)`.
   - **Details**: Based on POI distribution similarity.
   - **Availability**: NYC only (Chicago lacks POI data).

6. **grid_to_node_mapping.pkl**
   - **Purpose**: Mapping between grid cells and graph nodes.
   - **Shape**: `(W*H, N)`.
   - **Details**: Links grid data to graph nodes (binary or weighted).
   - **Use**: Enables grid-graph data integration.
### 📝 Notes
- NYC includes POI-related files and features (D=48), while Chicago does not. Specifically, Chicago's spatiotemporal_features.pkl lacks the POI dimensions (33–39), resulting in D=41 (features 0–40 only, followed by inflow and outflow at indices 41–42). Adjust model inputs accordingly when working with the Chicago dataset.
- Grid size `(W, H)` and node count `(N)` should be specified based on the specific dataset used.

---

## 🚀 Usage

#### Sample Code
To load and inspect the main dataset:
```python
import pickle
import numpy as np

# Load spatiotemporal_features.pkl
with open('spatiotemporal_features.pkl', 'rb') as f:
    data = pickle.load(f)

# Print shape
T, D, W, H = data.shape
print(f"Shape: T={T}, D={D}, W={W}, H={H}")

# Access risk values (feature 0)
risk = data[:, 0, :, :]
print("Risk shape:", risk.shape)

```
#### Example Workflow  
1. Load `spatiotemporal_features.pkl` to access spatiotemporal features.
2. Use `high_risk_mask.pkl` to focus on high-risk regions.
3. Leverage `risk_similarity_adjacency.pkl`, `road_network_adjacency.pkl`, and `poi_similarity_adjacency.pkl` for graph-based modeling (e.g., with Graph Neural Networks).
4. Map between grid and graph data using `grid_to_node_mapping.pkl`.



