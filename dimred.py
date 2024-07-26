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
from generate_images_grid import generate_embedding_images
import bones_utils

from bones_utils import *

class SkeletonEncoder:
    def __init__(self):
        self.skeletons = []  # List to store skeletons

        # self.upper_body_indices = [5, 6, 9, 10, 11, 12]
        self.upper_body_indices = [4, 5, 6]
        self.lower_body_indices = [11, 12, 13, 14, 15, 16]
        self.l_arm_body_indices = [5, 7, 9, 11]
        self.r_arm_body_indices = [6, 8, 10, 12]

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
        self.upper_embeddings = None
        self.lower_embeddings = None
        self.l_arm_embeddings = None
        self.r_arm_embeddings = None

        # boundaries for scalinga and normalization
        self.upper_min = None
        self.upper_max = None

        self.lower_min = None
        self.lower_max = None

        self.l_arm_min = None
        self.l_arm_max = None
        
        self.r_arm_min = None
        self.r_arm_max = None

        self.upper_body_range = []
        self.lower_body_range = []
        self.l_arm_range = []
        self.r_arm_range = []

        self.upper_angles = None
        self.lower_angles = None  

    def fit(self):
        n_neighbors = 300
        n_epochs = 1000
        min_dist = 0.25
        n_components = 2
        self.upper_model = umap_.UMAP(n_neighbors=n_neighbors, n_epochs=n_epochs, min_dist=min_dist, n_components=n_components)
        self.lower_model = umap_.UMAP(n_neighbors=n_neighbors, n_epochs=n_epochs, min_dist=min_dist, n_components=n_components)
        self.l_arm_model = umap_.UMAP(n_neighbors=n_neighbors, n_epochs=n_epochs, min_dist=min_dist, n_components=n_components)
        self.r_arm_model = umap_.UMAP(n_neighbors=n_neighbors, n_epochs=n_epochs, min_dist=min_dist, n_components=n_components)

        self.upper_embeddings = self.upper_model.fit_transform(np.array(self.upper_body_data).reshape(len(self.upper_body_data), -1))
        self.lower_embeddings = self.lower_model.fit_transform(np.array(self.lower_body_data).reshape(len(self.lower_body_data), -1))
        self.l_arm_embeddings = self.l_arm_model.fit_transform(np.array(self.l_arm_data).reshape(len(self.l_arm_data), -1))
        self.r_arm_embeddings = self.r_arm_model.fit_transform(np.array(self.r_arm_data).reshape(len(self.r_arm_data), -1))

        self.compute_index()


    def encode_fast_knn(self, skeleton):
        upper_points = np.array([skeleton[i] for i in self.upper_body_indices])
        lower_points = np.array([skeleton[i] for i in self.lower_body_indices])
        l_arm_points = np.array([skeleton[i] for i in self.l_arm_body_indices])
        r_arm_points = np.array([skeleton[i] for i in self.r_arm_body_indices])

        def find_closest_embedding_knn(embeddings, target_points, nn):

            target_points_flat = target_points.reshape(1, -1).astype(np.float32)  # Reshape and convert to float32
            distances, indices = nn.search(target_points_flat, k=10)  # Find the nearest neighbors
            
            # Calculate the weighted average of the positions based on distances
            nearest_positions = embeddings[indices[0]]
            weights = 1 / (distances[0] + 1e-5)  # Add a small value to avoid division by zero
            weighted_position = np.average(nearest_positions, axis=0, weights=weights)
        

            return weighted_position

        # Find closest embeddings
        upper_embedding = find_closest_embedding_knn(self.upper_embeddings, upper_points, self.upper_nn)
        lower_embedding = find_closest_embedding_knn(self.lower_embeddings, lower_points, self.lower_nn)
        l_arm_embedding = find_closest_embedding_knn(self.l_arm_embeddings, l_arm_points, self.l_arm_nn)
        r_arm_embedding = find_closest_embedding_knn(self.r_arm_embeddings, r_arm_points, self.r_arm_nn)

        return upper_embedding, lower_embedding, l_arm_embedding, r_arm_embedding

    # saving model
    def save(self, filename):
        if self.upper_embeddings is None or self.lower_embedding is None:
            raise RuntimeError("Must fit the models first using fit() method.")
        with open(filename, 'wb') as f:
            pickle.dump({
                'upper_model': self.upper_model,
                'lower_model': self.lower_model,
                'l_arm_model' : self.l_arm_model,
                'r_arm_model' : self.r_arm_model,
                'upper_embeddings': self.upper_embeddings,
                'lower_embeddings': self.lower_embeddings,
                'l_arm_embeddings': self.l_arm_embeddings,
                'r_arm_embeddings': self.r_arm_embeddings,
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

            self.upper_embeddings = data['upper_embeddings']
            self.lower_embeddings = data['lower_embeddings']
            self.l_arm_embeddings = data['l_arm_embeddings']
            self.r_arm_embeddings = data['r_arm_embeddings']

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
                                    
        self.upper_body_data = np.array(self.upper_body_data)
        self.lower_body_data = np.array(self.lower_body_data)
        self.l_arm_data = np.array(self.l_arm_data)
        self.r_arm_data = np.array(self.r_arm_data)

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
    encoder.fit()
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

