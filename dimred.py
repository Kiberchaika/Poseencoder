import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw
from umap import umap_
import time
import numpy as np
import faiss


from utils import *

class SkeletonEncoder:
    def __init__(self):
        self.skeletons = []  # List to store skeletons

        # self.upper_body_indices = [5, 6, 9, 10, 11, 12]
        self.upper_body_indices = [4, 5, 6]
        self.lower_body_indices = [11, 12, 13, 14, 15, 16]
        self.l_arm_body_indices = [5, 7, 9, 11]
        self.r_arm_body_indices = [6, 8, 10, 12]

        # Data to pass into fitting
        self.upper_body_data_norm = None
        self.lower_body_data_norm = None
        self.l_arm_data_norm = None
        self.r_arm_data_norm = None

        # Data that had been cut from skeletons
        self.upper_body_data = None
        self.lower_body_data = None
        self.l_arm_data = None
        self.r_arm_data = None

        # UMAP model's placeholder
        self.upper_model = None
        self.lower_model = None
        self.l_arm_model = None
        self.r_arm_model = None

        # embedding clouds
        self.upper_embedding = None
        self.lower_embedding = None
        self.l_arm_embedding = None
        self.r_arm_embedding = None

        # boundaries for scalinga and normalization
        self.upper_min = None
        self.upper_max = None

        self.lower_min = None
        self.lower_max = None

        self.l_arm_min = None
        self.l_arm_max = None
        
        self.r_arm_min = None
        self.r_arm_max = None

        # paths for corresponding images
        self.image_paths = []

        # represents pairs of vectors
        self.l_arm_angles = []
        self.r_arm_angles = []
        self.body_angles = []
        self.legs_angles = []

        self.upper_body_range = []
        self.lower_body_range = []
        self.l_arm_range = []
        self.r_arm_range = []

        self.upper_angles = None
        self.lower_angles = None

    # add skeleton to base. skeleton would be scaled, normalized adn only after that added
    def add(self, skeleton):
        # Scale the skeleton
        upper_points_scaled, lower_points_scaled, l_arm_points_scaled, r_arm_points_scaled = self.scale_skeleton(skeleton)

        # Normalize the keypoints
        up_norm, low_norm, l_arm_norm, r_arm_norm = self.normalize_keypoints(
            upper_points_scaled, lower_points_scaled,
            l_arm_points_scaled, r_arm_points_scaled
        )

        # Append scaled and normalized points to the lists
        if hasattr(self, 'upper_body_data') and self.upper_body_data.size:
            self.upper_body_data = np.concatenate((self.upper_body_data, [upper_points_scaled]), axis=0)
            self.lower_body_data = np.concatenate((self.lower_body_data, [lower_points_scaled]), axis=0)
            self.l_arm_data = np.concatenate((self.l_arm_data, [l_arm_points_scaled]), axis=0)
            self.r_arm_data = np.concatenate((self.r_arm_data, [r_arm_points_scaled]), axis=0)
        else:
            self.upper_body_data = np.array([upper_points_scaled])
            self.lower_body_data = np.array([lower_points_scaled])
            self.l_arm_data = np.array([l_arm_points_scaled])
            self.r_arm_data = np.array([r_arm_points_scaled])

        # Normalize and append the normalized points to the lists
        if hasattr(self, 'upper_body_data_norm') and self.upper_body_data_norm.size:
            self.upper_body_data_norm = np.concatenate((self.upper_body_data_norm, [up_norm]), axis=0)
            self.lower_body_data_norm = np.concatenate((self.lower_body_data_norm, [low_norm]), axis=0)
            self.l_arm_data_norm = np.concatenate((self.l_arm_data_norm, [l_arm_norm]), axis=0)
            self.r_arm_data_norm = np.concatenate((self.r_arm_data_norm, [r_arm_norm]), axis=0)
        else:
            self.upper_body_data_norm = np.array([up_norm])
            self.lower_body_data_norm = np.array([low_norm])
            self.l_arm_data_norm = np.array([l_arm_norm])
            self.r_arm_data_norm = np.array([r_arm_norm])

    def calculate_angles(self, points, body_indices):
        angles = []
        
        for bone in BONES_COCO:
            if bone[0] in body_indices and bone[1] in body_indices:
                index_1 = body_indices.index(bone[0])
                index_2 = body_indices.index(bone[1])
                v1 = points[index_1]
                v2 = points[index_2]
                angle = self.calculate_angle_between_vectors(v1, v2)
                angles.append(angle)

        return angles
        
    def scale_skeleton(self, skeleton):
        def scale_points(points, min_bound, max_bound):
            min_points, max_points = points.min(axis=0), points.max(axis=0)
            scale = min((max_bound - min_bound) / (max_points - min_points))
            scaled_points = (points - min_points) * scale + min_bound
            center_bound_x = (max_bound[0] + min_bound[0]) / 2
            hip_center_x = (scaled_points[:, 0].min() + scaled_points[:, 0].max()) / 2
            scaled_points[:, 0] += center_bound_x - hip_center_x
            return scaled_points

        upper_points = np.array([skeleton[i] for i in self.upper_body_indices])
        lower_points = np.array([skeleton[i] for i in self.lower_body_indices])
        l_arm_points = np.array([skeleton[i] for i in self.l_arm_body_indices])
        r_arm_points = np.array([skeleton[i] for i in self.r_arm_body_indices])

        upper_points_scaled = scale_points(upper_points, self.upper_min, self.upper_max)
        lower_points_scaled = scale_points(lower_points, self.lower_min, self.lower_max)
        l_arm_points_scaled = scale_points(l_arm_points, self.l_arm_min, self.l_arm_max)
        r_arm_points_scaled = scale_points(r_arm_points, self.r_arm_min, self.r_arm_max)

        return upper_points_scaled, lower_points_scaled, l_arm_points_scaled, r_arm_points_scaled


    def normalize_keypoints(self, upper_points, lower_points, l_arm_points, r_arm_points):
        """
        Normalize body data points between 0 and 1 using the specified min and max values.
        
        :param upper_points: Upper body keypoints to be normalized.
        :param lower_points: Lower body keypoints to be normalized.
        :param l_arm_points: Left arm keypoints to be normalized.
        :param r_arm_points: Right arm keypoints to be normalized.
        
        :return: Normalized upper body, lower body, left arm, and right arm keypoints.
        """
        # Check if min and max values are set
        if self.upper_min is None or self.upper_max is None:
            raise ValueError("Min and max values for upper body normalization are not set.")
        if self.lower_min is None or self.lower_max is None:
            raise ValueError("Min and max values for lower body normalization are not set.")
        if self.l_arm_min is None or self.l_arm_max is None:
            raise ValueError("Min and max values for left arm normalization are not set.")
        if self.r_arm_min is None or self.r_arm_max is None:
            raise ValueError("Min and max values for right arm normalization are not set.")
        
        # Normalize upper body data points
        upper_points_normalized = (upper_points - self.upper_min) / (self.upper_max - self.upper_min)

        # Normalize lower body data points
        lower_points_normalized = (lower_points - self.lower_min) / (self.lower_max - self.lower_min)
        
        # Normalize left arm data points
        l_arm_points_normalized = (l_arm_points - self.l_arm_min) / (self.l_arm_max - self.l_arm_min)
        
        # Normalize right arm data points
        r_arm_points_normalized = (r_arm_points - self.r_arm_min) / (self.r_arm_max - self.r_arm_min)

        return upper_points_normalized, lower_points_normalized, l_arm_points_normalized, r_arm_points_normalized

        
    # Normalize ALL data loaded to encoder
    def normalize_all_data_and_create_normalized_knn_cache(self):
        """
        Normalize upper, lower, and arm data points between 0 and 1.
        """
        def normalize(data):
            data_array = np.array(data).reshape(-1, 2)
            data_min = np.min(data_array, axis=0)
            data_max = np.max(data_array, axis=0)
            return np.array([(np.array(points) - data_min) / (data_max - data_min) for points in data]), data_min, data_max

        # Normalize and store the data
        self.upper_body_data_norm, self.upper_min, self.upper_max = normalize(self.upper_body_data)
        self.lower_body_data_norm, self.lower_min, self.lower_max = normalize(self.lower_body_data)
        self.l_arm_data_norm, self.l_arm_min, self.l_arm_max = normalize(self.l_arm_data)
        self.r_arm_data_norm, self.r_arm_min, self.r_arm_max = normalize(self.r_arm_data)
        


    def fit_with_normalized_data(self):
        self.upper_model = umap_.UMAP(n_components=2)
        self.lower_model = umap_.UMAP(n_components=2)
        self.l_arm_model = umap_.UMAP(n_components=2)
        self.r_arm_model = umap_.UMAP(n_components=2)

        # self.l_arm_angles
        # self.r_arm_angles
        # self.body_angles
        # self.legs_angles

        self.upper_embedding = self.upper_model.fit_transform(np.array(self.upper_body_data_norm).reshape(len(self.upper_body_data_norm), -1))
        self.lower_embedding = self.lower_model.fit_transform(np.array(self.lower_body_data_norm).reshape(len(self.lower_body_data_norm), -1))
        self.l_arm_embedding = self.l_arm_model.fit_transform(np.array(self.l_arm_data_norm).reshape(len(self.l_arm_data_norm), -1))
        self.r_arm_embedding = self.r_arm_model.fit_transform(np.array(self.r_arm_data_norm).reshape(len(self.r_arm_data_norm), -1))

    def fit_with_non_normalized_data(self):
        n_neighbors = 300
        n_epochs = 1000
        min_dist = 0.25
        n_components = 2
        self.upper_model = umap_.UMAP(n_neighbors=n_neighbors, n_epochs=n_epochs, min_dist=min_dist, n_components=n_components)
        self.lower_model = umap_.UMAP(n_neighbors=n_neighbors, n_epochs=n_epochs, min_dist=min_dist, n_components=n_components)
        self.l_arm_model = umap_.UMAP(n_neighbors=n_neighbors, n_epochs=n_epochs, min_dist=min_dist, n_components=n_components)
        self.r_arm_model = umap_.UMAP(n_neighbors=n_neighbors, n_epochs=n_epochs, min_dist=min_dist, n_components=n_components)

        # self.l_arm_angles
        # self.r_arm_angles
        # self.body_angles
        # self.legs_angles

        self.upper_embedding = self.upper_model.fit_transform(np.array(self.upper_body_data).reshape(len(self.upper_body_data), -1))
        self.lower_embedding = self.lower_model.fit_transform(np.array(self.lower_body_data).reshape(len(self.lower_body_data), -1))
        self.l_arm_embedding = self.l_arm_model.fit_transform(np.array(self.l_arm_data).reshape(len(self.l_arm_data), -1))
        self.r_arm_embedding = self.r_arm_model.fit_transform(np.array(self.r_arm_data).reshape(len(self.r_arm_data), -1))

        self.compute_index()
    
    def generate_embedding_images(self, 
                              upper_filename="upper_embeddings.png", 
                              lower_filename="lower_embeddings.png", 
                              l_arm_filename="l_arm_embeddings.png", 
                              r_arm_filename="r_arm_embeddings.png", 
                              grid_size=(20, 20), 
                              image_size=(32, 32), 
                              upper_2d_points=None, 
                              lower_2d_points=None, 
                              l_arm_2d_points=None, 
                              r_arm_2d_points=None, 
                              zoom_factor=2):
        """
        Create a large image with small images representing the embedding points arranged in a grid.
        Optionally include red dots representing the bounding rectangles of 2D keypoints.

        :param upper_filename: Filename for the upper body embeddings image.
        :param lower_filename: Filename for the lower body embeddings image.
        :param l_arm_filename: Filename for the left arm embeddings image.
        :param r_arm_filename: Filename for the right arm embeddings image.
        :param grid_size: The size of the grid (rows, columns).
        :param image_size: The size of each individual image in the grid.
        :param upper_2d_points: Optional list of 2D points for the upper body to include red dots.
        :param lower_2d_points: Optional list of 2D points for the lower body to include red dots.
        :param l_arm_2d_points: Optional list of 2D points for the left arm to include red dots.
        :param r_arm_2d_points: Optional list of 2D points for the right arm to include red dots.
        :param zoom_factor: Factor by which to zoom the images.
        """

        def draw_keypoints(draw, keypoints, indices, color='yellow', other_color='gray'):
            """Draw keypoints and bones on the image."""
            for i, j in BONES_COCO:
                if i in indices and j in indices:
                    draw.line([keypoints[i], keypoints[j]], fill=color, width=10)
                else:
                    draw.line([keypoints[i], keypoints[j]], fill=other_color, width=10)

        def create_grid_image(embedding, filename, points_2d=None, keypoint_indices=None):
            """Create a grid image for embedding points."""
            # Determine grid bounds
            x_min, y_min = np.min(embedding, axis=0)
            x_max, y_max = np.max(embedding, axis=0)

            # Create mesh grid
            x_bins = np.linspace(x_min, x_max, grid_size[0] + 1)
            y_bins = np.linspace(y_min, y_max, grid_size[1] + 1)

            # Create an empty image for the grid
            grid_image = Image.new('RGB', (grid_size[0] * image_size[0], grid_size[1] * image_size[1]), (0, 0, 0))
            draw = ImageDraw.Draw(grid_image)

            # Assign images to grid cells
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    x_start, x_end = x_bins[i], x_bins[i + 1]
                    y_start, y_end = y_bins[j], y_bins[j + 1]

                    # Find points within the current grid cell
                    points_in_cell = [k for k in range(len(embedding)) if x_start <= embedding[k, 0] < x_end and y_start <= embedding[k, 1] < y_end]

                    if points_in_cell:
                        # Choose a random point within the cell
                        chosen_index = random.choice(points_in_cell)
                        
                        # Create a black background image
                        img = Image.new('RGB', (640, 640), (0, 0, 0))

                        # Draw keypoints on the image
                        skeleton = self.skeletons[chosen_index]
                        keypoints = [(skeleton[idx][0], skeleton[idx][1]) for idx in range(len(skeleton))]
                        cell_draw = ImageDraw.Draw(img)
                        draw_keypoints(cell_draw, keypoints, keypoint_indices)

                        # Calculate zoomed size
                        zoomed_size = (int(image_size[0] * zoom_factor), int(image_size[1] * zoom_factor))
                        zoomed_img = img.resize(zoomed_size)

                        # Calculate crop box to center the zoomed image
                        left = (zoomed_size[0] - image_size[0]) / 2
                        top = (zoomed_size[1] - image_size[1]) / 2
                        right = (zoomed_size[0] + image_size[0]) / 2
                        bottom = (zoomed_size[1] + image_size[1]) / 2
                        crop_box = (left, top, right, bottom)

                        # Crop the zoomed image
                        cell_img = zoomed_img.crop(crop_box)

                        # Paste the cell image into the grid
                        grid_image.paste(cell_img, (i * image_size[0], j * image_size[1]))
                    else:
                        # Fill the cell with black if no points are in the cell
                        img = Image.new('RGB', image_size, (0, 0, 0))
                        grid_image.paste(img, (i * image_size[0], j * image_size[1]))

            # Draw red dots for 2D points if provided
            if points_2d is not None:
                # Calculate the position in the grid image
                grid_x = int((points_2d[0] - x_min) / (x_max - x_min) * grid_size[0] * image_size[0])
                grid_y = int((points_2d[1] - y_min) / (y_max - y_min) * grid_size[1] * image_size[1])
                # Draw a red dot
                draw.ellipse((grid_x - 5, grid_y - 5, grid_x + 5, grid_y + 5), fill='red')

            # Save the resulting image
            grid_image.save(filename)
            print(f"Image {filename} saved")

            return [x_min, y_min, x_max, y_max]

        # Generate images for upper, lower, left arm, and right arm embeddings
        self.upper_body_range = create_grid_image(self.upper_embedding, upper_filename, points_2d=upper_2d_points, keypoint_indices=self.upper_body_indices)
        self.lower_body_range = create_grid_image(self.lower_embedding, lower_filename, points_2d=lower_2d_points, keypoint_indices=self.lower_body_indices)
        self.l_arm_range = create_grid_image(self.l_arm_embedding, l_arm_filename, points_2d=l_arm_2d_points, keypoint_indices=self.l_arm_body_indices)
        self.r_arm_range = create_grid_image(self.r_arm_embedding, r_arm_filename, points_2d=r_arm_2d_points, keypoint_indices=self.r_arm_body_indices)

    
    def encode_fast_to_normalized_data(self, skeleton, filename="embeddings_with_points.png", draw=False):
        # Scale and normalize skeleton
        # upper_points, lower_points, l_arm_points, r_arm_points = self.scale_skeleton(skeleton)

        upper_points = np.array([skeleton[i] for i in self.upper_body_indices])
        lower_points = np.array([skeleton[i] for i in self.lower_body_indices])
        l_arm_points = np.array([skeleton[i] for i in self.l_arm_body_indices])
        r_arm_points = np.array([skeleton[i] for i in self.r_arm_body_indices])

        upper_points, lower_points, l_arm_points, r_arm_points = self.normalize_keypoints(
            upper_points, lower_points, l_arm_points, r_arm_points
        )

        def find_closest_embedding(data_norm, target_points, embeddings):
            # Convert to numpy array if it's a list
            target_points = np.array(target_points)

            # Compute the squared differences
            differences = data_norm - target_points[np.newaxis, :, :]

            # Compute the squared Euclidean distances
            squared_dists = np.sum(differences**2, axis=2)

            # Take the square root to get the Euclidean distances
            distances = np.sqrt(np.sum(squared_dists, axis=1))
            closest_index = np.argmin(distances)
            return embeddings[closest_index]

        # Find closest embeddings
        upper_embedding = find_closest_embedding(self.upper_body_data_norm, upper_points, self.upper_embedding)
        lower_embedding = find_closest_embedding(self.lower_body_data_norm, lower_points, self.lower_embedding)
        l_arm_embedding = find_closest_embedding(self.l_arm_data_norm, l_arm_points, self.l_arm_embedding)
        r_arm_embedding = find_closest_embedding(self.r_arm_data_norm, r_arm_points, self.r_arm_embedding)

        if draw:
            def plot_embedding(ax, embedding, color, label, title, highlight_embedding=None, highlight_color='yellow'):
                ax.scatter(embedding[:, 0], embedding[:, 1], c=color, label=label)
                if highlight_embedding is not None:
                    ax.scatter(highlight_embedding[0], highlight_embedding[1], marker='*', s=200, c=highlight_color, label='Current Point')
                ax.set_title(title)
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.legend()
                ax.invert_yaxis()

            plt.figure(figsize=(16, 8))

            plot_embedding(plt.subplot(2, 2, 1), self.upper_embedding, 'blue', 'Upper Body Embeddings', 'Upper Body UMAP Embeddings', upper_embedding, 'red')
            plot_embedding(plt.subplot(2, 2, 2), self.lower_embedding, 'red', 'Lower Body Embeddings', 'Lower Body UMAP Embeddings', lower_embedding, 'green')
            plot_embedding(plt.subplot(2, 2, 3), self.l_arm_embedding, 'gray', 'Left Arm Embeddings', 'Left Arm UMAP Embeddings', l_arm_embedding)
            plot_embedding(plt.subplot(2, 2, 4), self.r_arm_embedding, 'gray', 'Right Arm Embeddings', 'Right Arm UMAP Embeddings', r_arm_embedding)

            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

        return upper_embedding, lower_embedding, l_arm_embedding, r_arm_embedding

    def encode_fast(self, skeleton, filename="embeddings_with_points.png", draw=False):
        # Scale and normalize skeleton
        # upper_points, lower_points, l_arm_points, r_arm_points = self.scale_skeleton(skeleton)

        upper_points = np.array([skeleton[i] for i in self.upper_body_indices])
        lower_points = np.array([skeleton[i] for i in self.lower_body_indices])
        l_arm_points = np.array([skeleton[i] for i in self.l_arm_body_indices])
        r_arm_points = np.array([skeleton[i] for i in self.r_arm_body_indices])

        #upper_points, lower_points, l_arm_points, r_arm_points = self.normalize_keypoints(
        #    upper_points, lower_points, l_arm_points, r_arm_points
        #)

        def find_closest_embedding(data_norm, target_points, embeddings):
            # Convert to numpy array if it's a list
            target_points = np.array(target_points)

            # Compute the squared differences
            differences = data_norm - target_points[np.newaxis, :, :]

            # Compute the squared Euclidean distances
            squared_dists = np.sum(differences**2, axis=2)

            # Take the square root to get the Euclidean distances
            distances = np.sqrt(np.sum(squared_dists, axis=1))
            closest_index = np.argmin(distances)
            return embeddings[closest_index]

        # Find closest embeddings
        upper_embedding = find_closest_embedding(self.upper_body_data, upper_points, self.upper_embedding)
        lower_embedding = find_closest_embedding(self.lower_body_data, lower_points, self.lower_embedding)
        l_arm_embedding = find_closest_embedding(self.l_arm_data, l_arm_points, self.l_arm_embedding)
        r_arm_embedding = find_closest_embedding(self.r_arm_data, r_arm_points, self.r_arm_embedding)

        if draw:
            def plot_embedding(ax, embedding, color, label, title, highlight_embedding=None, highlight_color='yellow'):
                ax.scatter(embedding[:, 0], embedding[:, 1], c=color, label=label)
                if highlight_embedding is not None:
                    ax.scatter(highlight_embedding[0], highlight_embedding[1], marker='*', s=200, c=highlight_color, label='Current Point')
                ax.set_title(title)
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.legend()
                ax.invert_yaxis()

            plt.figure(figsize=(16, 8))

            plot_embedding(plt.subplot(2, 2, 1), self.upper_embedding, 'blue', 'Upper Body Embeddings', 'Upper Body UMAP Embeddings', upper_embedding, 'red')
            plot_embedding(plt.subplot(2, 2, 2), self.lower_embedding, 'red', 'Lower Body Embeddings', 'Lower Body UMAP Embeddings', lower_embedding, 'green')
            plot_embedding(plt.subplot(2, 2, 3), self.l_arm_embedding, 'gray', 'Left Arm Embeddings', 'Left Arm UMAP Embeddings', l_arm_embedding)
            plot_embedding(plt.subplot(2, 2, 4), self.r_arm_embedding, 'gray', 'Right Arm Embeddings', 'Right Arm UMAP Embeddings', r_arm_embedding)

            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

        return upper_embedding, lower_embedding, l_arm_embedding, r_arm_embedding

    def encode_fast_knn(self, skeleton, filename="embeddings_with_points.png", draw=False):
        # Scale and normalize skeleton
        # upper_points, lower_points, l_arm_points, r_arm_points = self.scale_skeleton(skeleton)

        upper_points = np.array([skeleton[i] for i in self.upper_body_indices])
        lower_points = np.array([skeleton[i] for i in self.lower_body_indices])
        l_arm_points = np.array([skeleton[i] for i in self.l_arm_body_indices])
        r_arm_points = np.array([skeleton[i] for i in self.r_arm_body_indices])

        #upper_points, lower_points, l_arm_points, r_arm_points = self.normalize_keypoints(
        #    upper_points, lower_points, l_arm_points, r_arm_points
        #)

        def find_closest_embedding_knn(embeddings, target_points, nn):

            target_points_flat = target_points.reshape(1, -1).astype(np.float32)  # Reshape and convert to float32
            distances, indices = nn.search(target_points_flat, k=10)  # Find the nearest neighbors
            
            # Calculate the weighted average of the positions based on distances
            nearest_positions = embeddings[indices[0]]
            weights = 1 / (distances[0] + 1e-5)  # Add a small value to avoid division by zero
            weighted_position = np.average(nearest_positions, axis=0, weights=weights)
        

            return weighted_position

        # Find closest embeddings
        upper_embedding = find_closest_embedding_knn(self.upper_embedding, upper_points, self.upper_nn)
        lower_embedding = find_closest_embedding_knn(self.lower_embedding, lower_points, self.lower_nn)
        l_arm_embedding = find_closest_embedding_knn(self.l_arm_embedding, l_arm_points, self.l_arm_nn)
        r_arm_embedding = find_closest_embedding_knn(self.r_arm_embedding, r_arm_points, self.r_arm_nn)

        if draw:
            def plot_embedding(ax, embedding, color, label, title, highlight_embedding=None, highlight_color='yellow'):
                ax.scatter(embedding[:, 0], embedding[:, 1], c=color, label=label)
                if highlight_embedding is not None:
                    ax.scatter(highlight_embedding[0], highlight_embedding[1], marker='*', s=200, c=highlight_color, label='Current Point')
                ax.set_title(title)
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.legend()
                ax.invert_yaxis()

            plt.figure(figsize=(16, 8))

            plot_embedding(plt.subplot(2, 2, 1), self.upper_embedding, 'blue', 'Upper Body Embeddings', 'Upper Body UMAP Embeddings', upper_embedding, 'red')
            plot_embedding(plt.subplot(2, 2, 2), self.lower_embedding, 'red', 'Lower Body Embeddings', 'Lower Body UMAP Embeddings', lower_embedding, 'green')
            plot_embedding(plt.subplot(2, 2, 3), self.l_arm_embedding, 'gray', 'Left Arm Embeddings', 'Left Arm UMAP Embeddings', l_arm_embedding)
            plot_embedding(plt.subplot(2, 2, 4), self.r_arm_embedding, 'gray', 'Right Arm Embeddings', 'Right Arm UMAP Embeddings', r_arm_embedding)

            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

        return upper_embedding, lower_embedding, l_arm_embedding, r_arm_embedding


    # # something is wrong here
    # def decode(self, upper_embedding, lower_embedding, filename="decoded_skeleton_plot.png", draw = False):
    #     if self.upper_model is None or self.lower_model is None:
    #         raise RuntimeError("Must initialize the models first using fit() method.")

    #     upper_points = np.array(self.upper_model.inverse_transform(upper_embedding)[0])
    #     lower_points = np.array(self.lower_model.inverse_transform(lower_embedding)[0])
    #     upper_points = upper_points.reshape(-1, 2)
    #     lower_points = lower_points.reshape(-1, 2)

    #     skeleton = np.zeros((17, 2))

    #     for i in range(0, len(upper_points)):
    #         skeleton[self.upper_body_indices[i]] = upper_points[i]
    #     for i in range(0, len(lower_points), 2):
    #         skeleton[self.lower_body_indices[i]] = lower_points[i]

    #     if draw:
    #         plt.figure(figsize=(6, 6))
    #         plt.scatter(skeleton[:, 0], skeleton[:, 1], c='blue', label='Skeleton Keypoints')
    #         plt.title('Reconstructed Skeleton Keypoints')
    #         plt.xlabel('X Coordinate')
    #         plt.ylabel('Y Coordinate')
    #         plt.gca().invert_yaxis()
    #         plt.legend()
    #         plt.grid(True)

    #         plt.savefig(filename)
    #         plt.close()

    #     return skeleton


    # saving model
    def save(self, filename):
        if self.upper_embedding is None or self.lower_embedding is None:
            raise RuntimeError("Must fit the models first using fit() method.")
        with open(filename, 'wb') as f:
            pickle.dump({
                'upper_model': self.upper_model,
                'lower_model': self.lower_model,
                'l_arm_model' : self.l_arm_model,
                'r_arm_model' : self.r_arm_model,
                'upper_embedding': self.upper_embedding,
                'lower_embedding': self.lower_embedding,
                'l_arm_embedding': self.l_arm_embedding,
                'r_arm_embedding': self.r_arm_embedding,
                'upper_min': self.upper_min,
                'upper_max': self.upper_max,
                'lower_min': self.lower_min,
                'lower_max': self.lower_max,
                'l_arm_min': self.l_arm_min,
                'l_arm_max': self.l_arm_max,
                'r_arm_min': self.r_arm_min,
                'r_arm_max': self.r_arm_max,
                'upper_angles': self.upper_angles,
                'lower_angles': self.lower_angles,
                # 'upper_body_data_norm': self.upper_body_data_norm,
                # 'lower_body_data_norm': self.lower_body_data_norm,
                # 'l_arm_data_norm': self.l_arm_data_norm,
                # 'r_arm_data_norm': self.r_arm_data_norm,
                'upper_body_data': self.upper_body_data,
                'lower_body_data': self.lower_body_data,
                'l_arm_data': self.l_arm_data,
                'r_arm_data': self.r_arm_data,
                'upper_body_range': self.upper_body_range,
                'lower_body_range': self.lower_body_range,
                'l_arm_range': self.l_arm_range,
                'r_arm_range': self.r_arm_range,
            }, f)

    def compute_index(self):
            self.upper_nn = faiss.IndexFlatL2(self.upper_body_data.shape[-1] * self.upper_body_data.shape[-2])  
            self.upper_nn.add(self.upper_body_data.reshape(self.upper_body_data.shape[0], -1))
            self.lower_nn = faiss.IndexFlatL2(self.lower_body_data.shape[-1] * self.lower_body_data.shape[-2])  
            self.lower_nn.add(self.lower_body_data.reshape(self.lower_body_data.shape[0], -1))
            self.l_arm_nn = faiss.IndexFlatL2(self.l_arm_data.shape[-1] * self.l_arm_data.shape[-2])  
            self.l_arm_nn.add(self.l_arm_data.reshape(self.l_arm_data.shape[0], -1))
            self.r_arm_nn = faiss.IndexFlatL2(self.r_arm_data.shape[-1] * self.r_arm_data.shape[-2])  
            self.r_arm_nn.add(self.r_arm_data.reshape(self.r_arm_data.shape[0], -1))


    # loading model
    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.upper_model = data['upper_model']
            self.lower_model = data['lower_model']
            self.l_arm_model = data['l_arm_model']
            self.r_arm_model = data['r_arm_model']

            self.upper_embedding = data['upper_embedding']
            self.lower_embedding = data['lower_embedding']
            self.l_arm_embedding = data['l_arm_embedding']
            self.r_arm_embedding = data['r_arm_embedding']

            # self.upper_body_data_norm = data['upper_body_data_norm']
            # self.lower_body_data_norm = data['lower_body_data_norm']
            # self.l_arm_data_norm = data['l_arm_data_norm']
            # self.r_arm_data_norm = data['r_arm_data_norm']

            self.upper_body_data = data['upper_body_data']
            self.lower_body_data = data['lower_body_data']
            self.l_arm_data = data['l_arm_data']
            self.r_arm_data = data['r_arm_data']

            self.compute_index()

            self.upper_min = data['upper_min']
            self.upper_max = data['upper_max']
            self.lower_min = data['lower_min']
            self.lower_max = data['lower_max']
            self.l_arm_min = data['l_arm_min']
            self.l_arm_max = data['l_arm_max']
            self.r_arm_min = data['r_arm_min']
            self.r_arm_max = data['r_arm_max']

            self.upper_angles = data['upper_angles']
            self.lower_angles = data['lower_angles']

            self.upper_body_range = data['upper_body_range']
            self.lower_body_range = data['lower_body_range']
            self.l_arm_range = data['l_arm_range']
            self.r_arm_range = data['r_arm_range']


    # load numpy skeletons and corresponding image paths. All data is normalized at the end
    def load_motions_data(self, collection_path):
        self.skeletons = []

        self.upper_body_data = []
        self.lower_body_data = []
        self.l_arm_data = []
        self.r_arm_data = []
        
        self.image_paths = []

        for collection_folder_name in sorted(os.listdir(collection_path)):
            collection_folder_path = os.path.join(collection_path, collection_folder_name)
            if os.path.isdir(collection_folder_path):
                for motion_folder in sorted(os.listdir(collection_folder_path)):
                    motion_folder_path = os.path.join(collection_folder_path, motion_folder)
                    if os.path.isdir(motion_folder_path):
                        keypoints_path = os.path.join(motion_folder_path, "all_keypoints.npy")
                        if os.path.exists(keypoints_path):
                            motion_list = np.load(keypoints_path)
                            
                            for idx, keypoints in enumerate(motion_list):
                                # skip frames
                                if idx % 50 != 0:
                                    continue

                                if not np.array_equal(keypoints, np.zeros((17, 2))):
                                    upper_points = [keypoints[i] for i in self.upper_body_indices]
                                    lower_points = [keypoints[i] for i in self.lower_body_indices]
                                    l_arm_points = [keypoints[i] for i in self.l_arm_body_indices]
                                    r_arm_points = [keypoints[i] for i in self.r_arm_body_indices]
                                    self.upper_body_data.append(upper_points)
                                    self.lower_body_data.append(lower_points)
                                    self.l_arm_data.append(l_arm_points)
                                    self.r_arm_data.append(r_arm_points)
                                    self.skeletons.append(keypoints)
                            
                            images_path = os.path.join(motion_folder_path, "neutral/images")
                            for single_image in sorted(os.listdir(images_path)):
                                image_path = os.path.join(images_path, single_image)
                                self.image_paths.append(image_path)
        
        self.upper_body_data = np.array(self.upper_body_data)
        self.lower_body_data = np.array(self.lower_body_data)
        self.l_arm_data = np.array(self.l_arm_data)
        self.r_arm_data = np.array(self.r_arm_data)
        self.image_paths = np.array(self.image_paths)

        # self.normalize_all_data_and_create_normalized_knn_cache()
        # self.calculate_all_rotations()


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


if __name__ == "__main__":

    # static pose
    still_keypoints = np.array([
         (350, 139), # nose
         (368, 116), # left_eye
         (331, 116), # right_eye
         (391, 131), # left_ear
         (310, 129), # right_ear
         (447, 238), # left_shoulder
         (240, 239), # right_shoulder
         (485, 338), # left_elbow
         (224, 398), # right_elbow
         (471, 530), # left_wrist
         (221, 531), # right_wrist
         (420, 500), # left_hip
         (270, 497), # right_hip
         (405, 725), # left_knee
         (293, 729), # right_knee
         (402, 940), # left_ankle
         (289, 932)] # right_ankle
         )
    
    tpose_keypoints = np.array([
        (155,42),
        (161,37),
        (148,36),
        (171,37),
        (143,39),
        (184,68),
        (130,68),
        (234,68),
        (80,66),
        (281,67),
        (25,70),
        (178,166),
        (125,161),
        (175,225),
        (131,228),
        (177,289),
        (134,289)])
    
    double_bicep = np.array([
        (55,18),
        (59,14),
        (51,13),
        (69,15),
        (48,15),
        (83,34),
        (35,34),
        (118,30),
        (6,35),
        (93,6),
        (25,10),
        (77,103),
        (44,103),
        (76,163),
        (45,168),
        (71,224),
        (59,226)])
    
    r_arm_up = np.array([
        (370,392),
        (400,371),
        (350,368),
        (435,370),
        (332,371),
        (466,468),
        (263,493),
        (559,306),
        (239,693),
        (598,130),
        (233,829),
        (467,826),
        (267,835),
        (411,1045),
        (325,1047),
        (381,1278),
        (333,1278)])

    current_kpts = r_arm_up

    # Create an instance of SkeletonEncoder with UMAP as the reduction method
    encoder = SkeletonEncoder()

    # Загрузка модели из файла
    # encoder.load("skeleton_encoder_model_umap.pkl")

    print("loading_data...")
    # Загрузка данных
    encoder.load_motions_data(collection_path="/media/k4_nas/admin/Киберчайка/Датасеты/BEDLAM/processed")
    # encoder.load_motions_data(collection_path="data")
    print("loading_data done")

    # encoder.encode_fast(np.zeros((17,2)), draw = False)

    print("fitting model...")
    # Фиттинг модели
    encoder.fit_with_non_normalized_data()
    print("fitting model done")

    # N_entries = 20000
    # print(f"sampling random {N_entries} entries ...")
    # encoder.sample_random_entries(N_entries)
    # print("sampling random entries done")

    # print("encoding...")
    # ts = time.time()
    # for _ in range(0, 1000):
    #     body, legs, l_arm, r_arm = encoder.encode_fast(current_kpts, draw = False)
    # print(f"time = {(time.time() - ts)}")
    # ts = time.time()
    # for _ in range(0, 1000):
    #     body, legs, l_arm, r_arm = encoder.encode_fast_knn(current_kpts, draw = False)
    # print(f"time = {(time.time() - ts)}")

    body, legs, l_arm, r_arm = encoder.encode_fast_knn(current_kpts, draw = False)
    
    # # # still_keypoints[:, 0] += 100
    # # Заенкоженные точки, отображаются на графике embeddings_with_points.png.

    # print("encoding...")
    # ts = time.time()
    # mas_len = 100
    # for _ in range(0, mas_len):
    #     body, legs, l_arm, r_arm = encoder.encode_fast(current_kpts, draw = False)
    # print(f"FPS = {mas_len/(time.time() - ts)}")
    # print("encoding done")

    # print("plotting reduced data ...")
    # # Эмбеддинги на графике
    # encoder.plot_reduced_data()
    # print("reduced data plotting done")


    # Декод пока кривой
    # # something is wrong here
    # encoder.decode(encoded_points_up, encoded_points_low, draw = True)
    
    # print("adding to encoder ...")
    # # Можно добавлять скелеты любого размера. Масштабирование и нормализация автоматическая
    # encoder.add(still_keypoints)
    # print("adding to encoder done")


    # # # Здесь смотрю графики склелетов
    # # # for i in range(0, 10):
    # plt.figure(figsize=(6, 6))
    # current = random.choice(encoder.upper_body_data_norm)
    # plt.scatter(current_kpts[:, 0], current_kpts[:, 1], c='blue', label='Skeleton Keypoints')
    # plt.title('Reconstructed Skeleton Keypoints')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.gca().invert_yaxis()
    # plt.legend()
    # plt.grid(True)
    # # plt.xlim(0, 1)
    # # plt.ylim(0, 1)
    # # plt.show()
    # plt.savefig(f"kek.jpg")
    # plt.close()
    print("printing pose clouds ...")
    # Таблицы менделеева и на них же точка последнего заенкоженного скелета.
    os.makedirs("emb", exist_ok=True)
    os.makedirs("emb/upper_emb", exist_ok=True)
    os.makedirs("emb/lower_emb", exist_ok=True)
    os.makedirs("emb/l_arm_emb", exist_ok=True)
    os.makedirs("emb/r_arm_emb", exist_ok=True)
    for i in range(0, 5):
        encoder.generate_embedding_images(f"emb/upper_emb/{i}.jpg", f"emb/lower_emb/{i}.jpg", f"emb/l_arm_emb/{i}.jpg", f"emb/r_arm_emb/{i}.jpg", grid_size=(16, 16), image_size=(64, 64),
                                          upper_2d_points = body,
                                          lower_2d_points = legs,
                                          l_arm_2d_points = l_arm,
                                          r_arm_2d_points = r_arm)
    print("printing pose clouds done")

    print("saving model ...")
    encoder.save("skeleton_encoder_model_umap.pkl")
    print("saving model done")

