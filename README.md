# IGNeuro: Neuron Firing Pattern Analyzer

## Overview

**IGNeuro** is a graphical user interface (GUI) tool designed to analyze and visualize neuron firing patterns using various information geometry and machine learning techniques. The primary purpose of this project is to apply these advanced techniques to arbitrary datasets of neuron firing patterns, enabling users to practice and explore their applications in computational neuroscience.

The tool provides several key analyses:
1. **Principal Component Analysis (PCA)**
2. **t-Distributed Stochastic Neighbor Embedding (t-SNE)**
3. **Autoencoder Visualization**
4. **Joint Probabilities Calculation**
5. **Fisher Information**
6. **Geodesic Analysis**

## Purpose

The goal of IGNeuro is to offer a practical application of information geometry techniques for analyzing neuron firing patterns. This includes understanding how neurons interact, identifying underlying patterns, and exploring relationships within neural data. The tool is designed for both educational and research purposes, allowing users to:
- Gain insights into neuronal behavior and interactions.
- Develop skills in applying advanced data analysis techniques.
- Explore various methods for dimensionality reduction and visualization.

## Techniques Used

### 1. **Principal Component Analysis (PCA)**

**Purpose**: PCA is used for dimensionality reduction by transforming the data into a new coordinate system where the greatest variances lie along the principal axes. This technique simplifies the data while retaining its most important features.

**Why Useful**:
- **Variance Representation**: PCA identifies the directions (principal components) along which the data varies the most, helping to visualize and interpret complex datasets.
- **Data Reduction**: Reduces the number of dimensions needed to represent the data, making it easier to visualize and analyze.
- **Pattern Identification**: Helps in identifying patterns and clusters within high-dimensional data.

**Visualization**: The PCA plot shows the data projected onto the first two principal components, revealing the major axes of variation and clustering within the dataset.

### 2. **t-Distributed Stochastic Neighbor Embedding (t-SNE)**

**Purpose**: t-SNE is a nonlinear dimensionality reduction technique that visualizes high-dimensional data in lower dimensions (2D or 3D) while preserving local similarities.

**Why Useful**:
- **Preserves Local Structure**: t-SNE is effective at preserving the local neighborhood structure of the data, which helps in visualizing fine-grained similarities and differences.
- **Cluster Detection**: Reveals intricate clusters and sub-groups in the data that might not be apparent in linear projections like PCA.
- **Non-linear Relationships**: Captures non-linear relationships between data points, providing a deeper understanding of complex datasets.

**Visualization**: The t-SNE plot displays data points in a lower-dimensional space, highlighting clusters and local relationships between neurons.

### 3. **Autoencoder Visualization**

**Purpose**: Autoencoders are neural networks used for learning compressed representations of data. They encode the input data into a lower-dimensional space and then reconstruct it.

**Why Useful**:
- **Feature Learning**: Autoencoders learn a compressed representation of the data, capturing the most important features.
- **Dimensionality Reduction**: Provides a learned, potentially non-linear reduction of dimensionality, revealing hidden structures.
- **Reconstruction Quality**: The ability to reconstruct the original data helps in evaluating how well the reduced representation captures essential features.

**Visualization**: The autoencoder plot visualizes data in a reduced space, showing how neurons' firing patterns are represented and reconstructed.

### 4. **Joint Probabilities Calculation**

**Purpose**: Calculates the joint probabilities of neuron firing patterns, providing insights into the likelihood of different firing state combinations.

**Why Useful**:
- **Relationship Understanding**: Helps in understanding how the firing of one neuron is related to the firing of another.
- **Pattern Analysis**: Reveals patterns in how neurons co-fire, which can be indicative of neural coding and interactions.

**Visualization**: The joint probabilities heatmap shows the likelihood of different firing state combinations between neurons.

### 5. **Fisher Information**

**Purpose**: Measures the amount of information that observable random variables (neuron firing) carry about an unknown parameter (firing rate).

**Why Useful**:
- **Information Measurement**: Indicates the precision of estimates for firing rates based on observed firing patterns.
- **Precision Analysis**: Higher Fisher Information values suggest more precise estimates and better understanding of firing rates.

**Visualization**: The Fisher Information plot displays the information content for each neuron, highlighting how well firing rates are estimated.

### 6. **Geodesic Analysis**

**Purpose**: Calculates the geodesic distance between empirical distributions of neuron firing patterns, using Euclidean distance as a proxy.

**Why Useful**:
- **Distribution Comparison**: Measures the difference between probability distributions, indicating how similar or distinct firing patterns are.
- **Similarity Assessment**: Helps in assessing how closely related different neuronal conditions are based on their firing distributions.

**Visualization**: The geodesic analysis plot shows the distance between distributions, illustrating how different firing patterns relate to each other.

## Usage

To use IGNeuro, follow these steps:
1. **Generate Random Patterns**: Create random neuron firing patterns.
2. **Plot Patterns**: Visualize the firing patterns of neurons.
3. **Perform Analyses**: Use the GUI to calculate joint probabilities, Fisher information, and geodesic distances.
4. **Visualize Results**: Use PCA, t-SNE, and autoencoder visualizations to explore and interpret the data.

## Contributing

Feel free to contribute to this project by submitting pull requests or opening issues. Your feedback and improvements are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For more detailed explanations on each technique, please refer to their respective sections above. Enjoy exploring neuron firing patterns with IGNeuro!
