import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.spatial.distance import euclidean
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

class NeuronFiringGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IGNeuro: Neuron Firing Pattern Analyzer")
        self.geometry("800x900")
        
        # GUI Elements
        self.num_neurons_label = tk.Label(self, text="Number of Neurons:")
        self.num_neurons_label.pack()
        self.num_neurons_entry = tk.Entry(self, width=10)
        self.num_neurons_entry.pack()
        self.num_neurons_entry.insert(0, '2')  # Default value
        
        self.generate_button = tk.Button(self, text="Generate Random Patterns", command=self.generate_random_patterns)
        self.generate_button.pack()
        
        self.result_label = tk.Label(self, text="", justify="left")
        self.result_label.pack()
        
        self.plot_button = tk.Button(self, text="Plot Patterns", command=self.plot_patterns)
        self.plot_button.pack()
        
        self.fisher_button = tk.Button(self, text="Fisher Information", command=self.calculate_fisher_information)
        self.fisher_button.pack()
        
        self.geodesic_button = tk.Button(self, text="Geodesic Analysis", command=self.geodesic_analysis)
        self.geodesic_button.pack()

        # New Buttons
        self.pca_button = tk.Button(self, text="PCA Visualization", command=self.pca_visualization)
        self.pca_button.pack()
        
        self.tsne_button = tk.Button(self, text="t-SNE Visualization", command=self.tsne_visualization)
        self.tsne_button.pack()
        
        self.autoencoder_button = tk.Button(self, text="Autoencoder Visualization", command=self.autoencoder_visualization)
        self.autoencoder_button.pack()
        
        self.neuron_data = []
        self.data_df = pd.DataFrame()

        self.tree = ttk.Treeview(self)
        self.tree.pack(fill=tk.BOTH, expand=True)

    def generate_random_patterns(self):
        try:
            num_neurons = int(self.num_neurons_entry.get())
            length = 50  # Length of the pattern
            
            self.neuron_data = []
            for _ in range(num_neurons):
                lambda_rate = np.random.uniform(0.1, 0.5)
                neuron_data = np.random.poisson(lambda_rate, length)
                neuron_data = (neuron_data > 0).astype(int)
                self.neuron_data.append(neuron_data)
            
            messagebox.showinfo("Success", f"Random patterns generated for {num_neurons} neurons successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def calculate_probabilities(self):
        try:
            if len(self.neuron_data) < 2:
                raise ValueError("At least 2 neurons are required to calculate probabilities.")
            
            neuron1_data = self.neuron_data[0]
            neuron2_data = self.neuron_data[1]
            
            n = len(neuron1_data)
            n00 = sum(1 for x, y in zip(neuron1_data, neuron2_data) if x == 0 and y == 0)
            n01 = sum(1 for x, y in zip(neuron1_data, neuron2_data) if x == 0 and y == 1)
            n10 = sum(1 for x, y in zip(neuron1_data, neuron2_data) if x == 1 and y == 0)
            n11 = sum(1 for x, y in zip(neuron1_data, neuron2_data) if x == 1 and y == 1)
            
            p00 = n00 / n
            p01 = n01 / n
            p10 = n10 / n
            p11 = n11 / n
            
            result_text = (
                f"Joint Probabilities:\n"
                f"P(00) = {p00:.4f}\n"
                f"P(01) = {p01:.4f}\n"
                f"P(10) = {p10:.4f}\n"
                f"P(11) = {p11:.4f}\n\n"
                f"Explanation:\n"
                f"P(00): Probability that neither neuron fires.\n"
                f"P(01): Probability that Neuron 2 fires but Neuron 1 does not.\n"
                f"P(10): Probability that Neuron 1 fires but Neuron 2 does not.\n"
                f"P(11): Probability that both neurons fire."
            )
            
            self.result_label.config(text=result_text)
            self.visualize_joint_probabilities(p00, p01, p10, p11)
            
            self.save_results_to_csv('Joint Probabilities', [p00, p01, p10, p11])
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_patterns(self):
        try:
            if not self.neuron_data:
                raise ValueError("No neuron data to plot. Please generate random patterns first.")
            
            num_neurons = len(self.neuron_data)
            rows = min(num_neurons, 10)
            cols = int(np.ceil(num_neurons / rows))
            plt.figure(figsize=(15, rows * 2))
            
            for i, neuron_data in enumerate(self.neuron_data):
                plt.subplot(rows, cols, i + 1)
                plt.stem(neuron_data, linefmt='-', markerfmt='o', basefmt='-', use_line_collection=True)
                plt.ylim(-0.5, 1.5)
                plt.yticks([0, 1])
                plt.title(f'Neuron {i+1} Firing Pattern')
                plt.xlabel('Time')
                plt.ylabel('Firing')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def calculate_fisher_information(self):
        try:
            if not self.neuron_data:
                raise ValueError("No neuron data to analyze. Please generate random patterns first.")
            
            fisher_info_values = []
            for neuron_data in self.neuron_data:
                n = len(neuron_data)
                p = sum(neuron_data) / n
                fisher_info = n / (p * (1 - p))
                fisher_info_values.append(fisher_info)
            
            result_text = "Fisher Information:\n"
            for i, fisher_info in enumerate(fisher_info_values):
                result_text += f"Neuron {i+1}: {fisher_info:.4f}\n"
            result_text += "\nExplanation:\nFisher Information measures the amount of information that an observable random variable (neuron firing) carries about an unknown parameter (firing rate).\nHigher values indicate more precise estimates of the firing rate."
            
            self.result_label.config(text=result_text)
            self.visualize_fisher_information(fisher_info_values)
            
            self.save_results_to_csv('Fisher Information', fisher_info_values)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def visualize_fisher_information(self, fisher_info_values):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, fisher_info in enumerate(fisher_info_values):
            ax.bar3d([i+1], [i+1], [0], [1], [1], [fisher_info], alpha=0.7)
        ax.set_xlabel('Neuron')
        ax.set_ylabel('Neuron')
        ax.set_zlabel('Fisher Information')
        plt.title('Fisher Information for Neurons')
        ax.text2D(0.05, 0.95, "Fisher Information measures the amount of information that an observable random variable (neuron firing) carries about an unknown parameter (firing rate).\nHigher values indicate more precise estimates of the firing rate.", transform=ax.transAxes)
        plt.show()
    
    def visualize_joint_probabilities(self, p00, p01, p10, p11):
        fig, ax = plt.subplots()
        joint_probs = np.array([[p00, p01], [p10, p11]])
        cax = ax.matshow(joint_probs, cmap='viridis')
        fig.colorbar(cax)
        for (i, j), val in np.ndenumerate(joint_probs):
            ax.text(j, i, f'{val:.4f}', ha='center', va='center', color='white')
        ax.set_xticklabels(['', 'Neuron 2 Not Firing', 'Neuron 2 Firing'])
        ax.set_yticklabels(['', 'Neuron 1 Not Firing', 'Neuron 1 Firing'])
        plt.title('Joint Probabilities')
        plt.figtext(0.5, 0.01, "Joint Probabilities represent the likelihood of combinations of neuron firing states.\nHigher probabilities indicate more frequent occurrences of the respective firing states.", wrap=True, horizontalalignment='center', fontsize=12)
        plt.show()

    def geodesic_analysis(self):
        try:
            if len(self.neuron_data) < 2:
                raise ValueError("At least 2 neurons are required for geodesic analysis.")
            
            # Define two different conditions for comparison
            condition1 = self.neuron_data[0]
            condition2 = self.neuron_data[1]
            
            # Calculate the empirical distributions
            p1 = np.bincount(condition1, minlength=2) / len(condition1)
            p2 = np.bincount(condition2, minlength=2) / len(condition2)
            
            # Calculate the geodesic distance (using Euclidean distance as a proxy)
            geodesic_distance = euclidean(p1, p2)
            
            result_text = (
                f"Geodesic Analysis:\n"
                f"Geodesic Distance between conditions: {geodesic_distance:.4f}\n\n"
                f"Explanation:\n"
                f"Geodesic Distance represents the most efficient transformation between two probability distributions.\n"
                f"Smaller values indicate more similar distributions, while larger values suggest more distinct firing patterns."
            )
            
            self.result_label.config(text=result_text)
            self.visualize_geodesic_analysis(p1, p2, geodesic_distance)
            
            self.save_results_to_csv('Geodesic Distance', [geodesic_distance])
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def visualize_geodesic_analysis(self, p1, p2, geodesic_distance):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot([0, 1], [0, 1], zs=0, zdir='z', label='curve in (x,y)')
        ax.plot([0, p1[1]], [0, p2[1]], zs=0, zdir='z', label='Geodesic Path', color='g')
        ax.scatter([0, p1[1]], [0, p2[1]], zs=[0, 0], zdir='z', color='r')
        ax.set_xlabel('Neuron 1 Probability')
        ax.set_ylabel('Neuron 2 Probability')
        ax.set_zlabel('Geodesic Distance')
        plt.title('Geodesic Distance Visualization')
        ax.text2D(0.05, 0.95, f"Geodesic Distance represents the most efficient transformation between two probability distributions.\nSmaller values indicate more similar distributions, while larger values suggest more distinct firing patterns.\nGeodesic Distance: {geodesic_distance:.4f}", transform=ax.transAxes)
        plt.show()

    def pca_visualization(self):
        try:
            if not self.neuron_data:
                raise ValueError("No neuron data to analyze. Please generate random patterns first.")
            
            data = np.array(self.neuron_data).T
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            pca = PCA(n_components=2)
            data_pca = pca.fit_transform(data_scaled)
            
            plt.figure(figsize=(8, 6))
            plt.scatter(data_pca[:, 0], data_pca[:, 1], c='blue', edgecolor='k', s=50)
            plt.title("PCA Visualization of Neuron Firing Patterns")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.grid(True)
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def tsne_visualization(self):
        try:
            if not self.neuron_data:
                raise ValueError("No neuron data to analyze. Please generate random patterns first.")
            
            data = np.array(self.neuron_data).T
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            tsne = TSNE(n_components=2, random_state=0)
            data_tsne = tsne.fit_transform(data_scaled)
            
            plt.figure(figsize=(8, 6))
            plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c='green', edgecolor='k', s=50)
            plt.title("t-SNE Visualization of Neuron Firing Patterns")
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
            plt.grid(True)
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def autoencoder_visualization(self):
        try:
            if not self.neuron_data:
                raise ValueError("No neuron data to analyze. Please generate random patterns first.")
            
            data = np.array(self.neuron_data)
            input_dim = data.shape[1]
            
            # Build autoencoder model
            input_layer = Input(shape=(input_dim,))
            encoded = Dense(2, activation='relu')(input_layer)
            decoded = Dense(input_dim, activation='sigmoid')(encoded)
            
            autoencoder = Model(input_layer, decoded)
            encoder = Model(input_layer, encoded)
            
            autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
            autoencoder.fit(data, data, epochs=50, batch_size=10, shuffle=True, verbose=0)
            
            encoded_data = encoder.predict(data)
            
            plt.figure(figsize=(8, 6))
            plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c='purple', edgecolor='k', s=50)
            plt.title("Autoencoder Visualization of Neuron Firing Patterns")
            plt.xlabel("Encoded Feature 1")
            plt.ylabel("Encoded Feature 2")
            plt.grid(True)
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_results_to_csv(self, analysis_type, values):
        results = {f"{analysis_type}": values}
        results_df = pd.DataFrame(results)
        self.data_df = pd.concat([self.data_df, results_df], axis=1)
        self.data_df.to_csv("neuron_analysis_results.csv", index=False)
        self.display_results_in_table()

    def display_results_in_table(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        
        self.tree["column"] = list(self.data_df.columns)
        self.tree["show"] = "headings"
        
        for col in self.tree["column"]:
            self.tree.heading(col, text=col)

        df_rows = self.data_df.to_numpy().tolist()
        for row in df_rows:
            self.tree.insert("", "end", values=row)

if __name__ == "__main__":
    app = NeuronFiringGUI()
    app.mainloop()
