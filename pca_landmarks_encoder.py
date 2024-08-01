import numpy as np
from sklearn.decomposition import PCA
from abstract_landmarks_encoder import BaseLandmarksEncoder
import torch

class PCALandmarksEncoder(BaseLandmarksEncoder):
    def __init__(self, input_landmarks_shape, embedding_shape):
        super().__init__(input_landmarks_shape, embedding_shape)
        self.pca_model = None
        self.mean = None
        self.min_values = None
        self.max_values = None

    def add(self, landmarks):
        if landmarks.shape != self.input_landmarks_shape:
            raise ValueError(f"Expected shape {self.input_landmarks_shape}, got {landmarks.shape}")
        self.data.append(landmarks)

    def fit(self):
        data_array = np.array(self.data).reshape(len(self.data), -1)
        self.pca_model = PCA(n_components=self.embedding_shape[0])
        embeddings = self.pca_model.fit_transform(data_array)
        self.mean = self.pca_model.mean_
        self.min_values = np.min(embeddings, axis=0)
        self.max_values = np.max(embeddings, axis=0)

    def encode(self, landmarks):
        if isinstance(landmarks, np.ndarray):
            landmarks_flat = landmarks.reshape(1, -1)
            embedding = self.pca_model.transform(landmarks_flat)
            normalized_embedding = (embedding - self.min_values) / (self.max_values - self.min_values)
            return normalized_embedding.squeeze(0)
        elif isinstance(landmarks, torch.Tensor):
            landmarks_flat = landmarks.view(1, -1)
            embedding = torch.from_numpy(self.pca_model.transform(landmarks_flat.cpu().numpy())).to(landmarks.device)
            normalized_embedding = (embedding - torch.from_numpy(self.min_values).to(landmarks.device)) / (torch.from_numpy(self.max_values - self.min_values).to(landmarks.device))
            return normalized_embedding.squeeze(0)
        else:
            raise TypeError("Input must be either a numpy array or a PyTorch tensor")

    def decode(self, normalized_encoding):
        if isinstance(normalized_encoding, np.ndarray):
            if normalized_encoding.ndim == 1:
                normalized_encoding = normalized_encoding[np.newaxis, :]
            encoding = normalized_encoding * (self.max_values - self.min_values) + self.min_values
            return self.pca_model.inverse_transform(encoding).reshape(self.input_landmarks_shape)
        elif isinstance(normalized_encoding, torch.Tensor):
            if normalized_encoding.dim() == 1:
                normalized_encoding = normalized_encoding.unsqueeze(0)
            encoding = normalized_encoding * torch.from_numpy(self.max_values - self.min_values).to(normalized_encoding.device) + torch.from_numpy(self.min_values).to(normalized_encoding.device)
            return torch.from_numpy(self.pca_model.inverse_transform(encoding.cpu().numpy())).to(normalized_encoding.device).view(self.input_landmarks_shape)
        else:
            raise TypeError("Input must be either a numpy array or a PyTorch tensor")

    def save(self, filename):
        np.savez(filename, 
                 components=self.pca_model.components_,
                 mean=self.pca_model.mean_,
                 explained_variance=self.pca_model.explained_variance_,
                 min_values=self.min_values,
                 max_values=self.max_values,
                 input_landmarks_shape=self.input_landmarks_shape,
                 embedding_shape=self.embedding_shape)

    def load(self, filename):
        with np.load(filename) as data:
            self.pca_model = PCA(n_components=data['components'].shape[0])
            self.pca_model.components_ = data['components']
            self.pca_model.mean_ = data['mean']
            self.pca_model.explained_variance_ = data['explained_variance']
            self.min_values = data['min_values']
            self.max_values = data['max_values']
            self.input_landmarks_shape = tuple(data['input_landmarks_shape'])
            self.embedding_shape = tuple(data['embedding_shape'])