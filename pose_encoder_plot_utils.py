import matplotlib.pyplot as plt

def plot_reduced_data(self, filename="reduced_data_plot.png"):
    if any(embedding is None for embedding in [self.upper_embedding, self.lower_embedding, self.l_arm_embedding, self.r_arm_embedding]):
        raise RuntimeError("Must fit the models first using fit() method.")

    embeddings = [
        (self.upper_embedding, 'Upper Body', 'blue'),
        (self.lower_embedding, 'Lower Body', 'red'),
        (self.l_arm_embedding, 'Left Arm', 'green'),
        (self.r_arm_embedding, 'Right Arm', 'purple')
    ]

    plt.figure(figsize=(24, 12))

    for i, (embedding, title, color) in enumerate(embeddings, 1):
        plt.subplot(2, 2, i)
        plt.scatter(embedding[:, 0], embedding[:, 1], c=color, label=title)
        plt.title(f'{title} UMAP Embedding')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def sample_random_entries(self, N):
    if N > self.upper_body_data_norm.shape[0]:
        raise ValueError("N cannot be greater than the number of elements in the first dimension of the array")

    # Select N random indices along the first dimension
    random_indices = np.random.choice(self.upper_body_data_norm.shape[0], N, replace=False)

    # Data to pass into fitting
    self.upper_body_data_norm = self.upper_body_data_norm[random_indices]
    self.lower_body_data_norm = self.lower_body_data_norm[random_indices]
    self.l_arm_data_norm = self.l_arm_data_norm[random_indices]
    self.r_arm_data_norm = self.r_arm_data_norm[random_indices]

    # Data that had been cut from skeletons
    self.upper_body_data = self.upper_body_data[random_indices]
    self.lower_body_data = self.lower_body_data[random_indices]
    self.l_arm_data = self.l_arm_data[random_indices]
    self.r_arm_data = self.r_arm_data[random_indices]

    # embedding clouds
    self.upper_embedding = self.upper_embedding[random_indices]
    self.lower_embedding = self.lower_embedding[random_indices]
    self.l_arm_embedding = self.l_arm_embedding[random_indices]
    self.r_arm_embedding = self.r_arm_embedding[random_indices]

    # paths for corresponding images
    self.image_paths = self.image_paths[random_indices]

def plot_embedding(ax, embedding, color, label, title, highlight_embedding=None, highlight_color='yellow'):
    ax.scatter(embedding[:, 0], embedding[:, 1], c=color, label=label)
    if highlight_embedding is not None:
        ax.scatter(highlight_embedding[0], highlight_embedding[1], marker='*', s=200, c=highlight_color, label='Current Point')
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend()
    ax.invert_yaxis()

def plot_four_embeddings(full_embedding, upper_embedding, lower_embedding, l_arm_embedding, r_arm_embedding, picture_output_filename="embeddings_with_points.png"):
    plt.figure(figsize=(16, 8))

    plot_embedding(plt.subplot(2, 2, 1), full_embedding, 'blue', 'Upper Body Embeddings', 'Upper Body UMAP Embeddings', upper_embedding, 'red')
    plot_embedding(plt.subplot(2, 2, 2), full_embedding, 'red', 'Lower Body Embeddings', 'Lower Body UMAP Embeddings', lower_embedding, 'green')
    plot_embedding(plt.subplot(2, 2, 3), full_embedding, 'gray', 'Left Arm Embeddings', 'Left Arm UMAP Embeddings', l_arm_embedding)
    plot_embedding(plt.subplot(2, 2, 4), full_embedding, 'gray', 'Right Arm Embeddings', 'Right Arm UMAP Embeddings', r_arm_embedding)

    plt.tight_layout()
    plt.savefig(picture_output_filename)
    plt.close()
