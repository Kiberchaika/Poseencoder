import numpy as np
import pickle
from umap.umap_ import UMAP
import faiss
from abstract_landmarks_encoder import BaseLandmarksEncoder
from scipy.optimize import minimize

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

    def encode(self, landmarks, triangulation=True):
        landmarks_flat = landmarks.reshape(1, -1).astype(np.float32)
        k = 10
        distances, indices = self.faiss_index.search(landmarks_flat, k=k)
        nearest_embeddings = self.embeddings[indices[0]]

        if triangulation:
            weights = self._optimize_weights(nearest_embeddings)
        else:
            weights = 1 / (distances[0] + 1e-5)
            weights /= np.sum(weights)

        return np.dot(weights, nearest_embeddings)

    def _optimize_weights(self, nearest_embeddings):
        def objective(weights):
            return np.sum((np.dot(weights, nearest_embeddings) - nearest_embeddings)**2)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [(0, 1) for _ in range(len(nearest_embeddings))]
        result = minimize(objective, x0=np.ones(len(nearest_embeddings))/len(nearest_embeddings),
                          method='SLSQP', constraints=constraints, bounds=bounds)
        return result.x
    
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