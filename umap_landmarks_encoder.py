import numpy as np
import pickle
from umap import UMAP
import faiss
from abstract_landmarks_encoder import BaseLandmarksEncoder

class UMAPLandmarksEncoder(BaseLandmarksEncoder):
    def __init__(self, input_landmarks_shape, embedding_shape):
        super().__init__(input_landmarks_shape, embedding_shape)
        self.umap_model = None
        self.embeddings = None
        self.faiss_index = None

    def add(self, landmarks):
        if landmarks.shape != self.input_landmarks_shape:
            raise ValueError(f"Expected shape {self.input_landmarks_shape}, got {landmarks.shape}")
        self.data.append(landmarks)

    def fit(self):
        data_array = np.array(self.data).reshape(len(self.data), -1)
        self.umap_model = UMAP(n_components=self.embedding_shape[0], n_neighbors=300, n_epochs=1000, min_dist=0.25)
        self.embeddings = self.umap_model.fit_transform(data_array)
        
        # Create FAISS index
        self.faiss_index = faiss.IndexFlatL2(data_array.shape[1])
        self.faiss_index.add(data_array.astype(np.float32))

    def encode(self, landmarks):
        if landmarks.shape != self.input_landmarks_shape:
            raise ValueError(f"Expected shape {self.input_landmarks_shape}, got {landmarks.shape}")
        
        landmarks_flat = landmarks.reshape(1, -1).astype(np.float32)
        distances, indices = self.faiss_index.search(landmarks_flat, k=10)
        
        nearest_embeddings = self.embeddings[indices[0]]
        weights = 1 / (distances[0] + 1e-5)
        weighted_embedding = np.average(nearest_embeddings, axis=0, weights=weights)
        
        return weighted_embedding

    def decode(self, encoding):
        # This method is not implemented in the original code
        # You might want to implement it in the future
        raise NotImplementedError("Decode method is not implemented yet")

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'umap_model': self.umap_model,
                'embeddings': self.embeddings,
                'data': self.data,
                'input_landmarks_shape': self.input_landmarks_shape,
                'embedding_shape': self.embedding_shape
            }, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.umap_model = data['umap_model']
            self.embeddings = data['embeddings']
            self.data = data['data']
            self.input_landmarks_shape = data['input_landmarks_shape']
            self.embedding_shape = data['embedding_shape']
        
        # Recreate FAISS index
        data_array = np.array(self.data).reshape(len(self.data), -1)
        self.faiss_index = faiss.IndexFlatL2(data_array.shape[1])
        self.faiss_index.add(data_array.astype(np.float32))