from abc import ABC, abstractmethod
import numpy as np

class BaseLandmarksEncoder(ABC):
    def __init__(self, input_landmarks_shape, embedding_shape):
        self.input_landmarks_shape = input_landmarks_shape
        self.embedding_shape = embedding_shape
        self.data = []

    @abstractmethod
    def add(self, landmarks):
        """Add landmarks to the encoder's dataset."""
        pass

    @abstractmethod
    def fit(self):
        """Fit the encoder to the added data."""
        pass

    @abstractmethod
    def encode(self, landmarks):
        """Encode the given landmarks."""
        pass

    @abstractmethod
    def decode(self, encoding):
        """Decode the given encoding back to landmarks."""
        pass

    def save(self, filename):
        """Save the encoder to a file."""
        pass

    def load(self, filename):
        """Load the encoder from a file."""
        pass