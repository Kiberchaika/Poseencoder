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