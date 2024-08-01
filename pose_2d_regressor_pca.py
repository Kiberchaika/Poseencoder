import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from regression_dataset_pca import PosesDatasetPCA
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import io
import os
from tqdm import tqdm
from datetime import datetime
import pickle

BONES_COCO = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (5, 6), # Shoulders
    (5, 7), (7, 9), # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    (0, 5), (0, 6)
]




class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class PoseRegressor(nn.Module):
    def __init__(self, input_dim=34, output_dim=8, input_projection_dim=128):
        super(PoseRegressor, self).__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     ResidualBlock(128, 128),
        #     nn.Linear(128, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     ResidualBlock(64, 64),
        #     nn.Linear(64, 32),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Linear(32, output_dim)
        # )
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_projection_dim),
            nn.BatchNorm1d(input_projection_dim),
            nn.ReLU(),
            ResidualBlock(input_projection_dim, 128),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResidualBlock(64, 64),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

   
    def forward(self, x):
        return self.model(x)

class FastKNN:
    def __init__(self, k=1):
        self.k = k
        self.embeddings = None
        self.skeletons = None

    def add(self, embeddings, skeletons):
        self.embeddings = embeddings
        self.skeletons = skeletons

    def find_nearest(self, query):
        with torch.no_grad():
            distances = torch.cdist(query.unsqueeze(0), self.embeddings)
            _, indices = distances.topk(self.k, largest=False, dim=1)
        return self.skeletons[indices[0]], self.embeddings[indices[0]]
    
class PoseRegressionModel:
    def __init__(self, encoders, input_dim=34, output_dim=8, learning_rate=0.001):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.encoders = encoders
        self.model = PoseRegressor(input_dim, output_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.modelname = f"REG_PCA_{current_time};"
        self.writer = SummaryWriter(f'poseregressor_pca_runs/{self.modelname}')
        self.knn = FastKNN()
        self.reg_weight = 0.001

        self.upper_body_indices = [4, 5, 6]
        self.lower_body_indices = [11, 12, 13, 14, 15, 16]
        self.l_arm_body_indices = [5, 7, 9, 11]
        self.r_arm_body_indices = [6, 8, 10, 12]

        # self.part_indices = {
        #     'upper': self.upper_body_indices,
        #     'lower': self.lower_body_indices,
        #     'l_arm': self.l_arm_body_indices,
        #     'r_arm': self.r_arm_body_indices
        # }
        # self.part_output_indices = {
        #     'upper': [0, 1],
        #     'lower': [2, 3],
        #     'l_arm': [4, 5],
        #     'r_arm': [6, 7]
        # }


    def plot_and_compare_skeletons(self, skeleton, nearest_skeleton, predicted_embeddings, target, step):
        fig, axs = plt.subplots(2, 2, figsize=(20, 20), dpi=100)
        fig.suptitle('Pose Analysis', fontsize=24)

        # Helper function to add index labels
        def add_index_labels(ax, points):
            for i, (x, y) in enumerate(points):
                ax.text(x, y, str(i), fontsize=10, ha='center', va='center', 
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        # Plot 1: Original 2D Skeleton
        axs[0, 0].scatter(skeleton[:, 0], skeleton[:, 1], c='b', s=50)
        for start, end in BONES_COCO:
            axs[0, 0].plot([skeleton[start, 0], skeleton[end, 0]],
                        [skeleton[start, 1], skeleton[end, 1]], 'r-', linewidth=2)
        add_index_labels(axs[0, 0], skeleton)
        axs[0, 0].set_title('Original 2D Skeleton', fontsize=18)
        axs[0, 0].set_aspect('equal')
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])

        # Plot 2: Nearest Neighbor 2D Skeleton
        axs[0, 1].scatter(nearest_skeleton[:, 0], nearest_skeleton[:, 1], c='b', s=50)
        for start, end in BONES_COCO:
            axs[0, 1].plot([nearest_skeleton[start, 0], nearest_skeleton[end, 0]],
                        [nearest_skeleton[start, 1], nearest_skeleton[end, 1]], 'r-', linewidth=2)
        add_index_labels(axs[0, 1], nearest_skeleton)
        axs[0, 1].set_title('Nearest Neighbor 2D Skeleton', fontsize=18)
        axs[0, 1].set_aspect('equal')
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])

        # Plot 3: PCA Coordinates
        colors = ['r', 'g', 'b', 'y']
        labels = ['upper', 'lower', 'l_arm', 'r_arm']

        # Assuming you have access to ground truth embeddings
        ground_truth_embeddings = target.cpu().numpy()  # Adjust this line based on how you store ground truth

        for i, (color, label) in enumerate(zip(colors, labels)):
            pred_x, pred_y = predicted_embeddings[i*2], predicted_embeddings[i*2+1]
            true_x, true_y = ground_truth_embeddings[i*2], ground_truth_embeddings[i*2+1]
            
            # Plot predicted point (larger)
            axs[1, 0].scatter(pred_x, pred_y, c=color, s=300, label=f'{label} (Predicted)', marker='o')
            
            # Plot ground truth point (smaller)
            axs[1, 0].scatter(true_x, true_y, c=color, s=100, label=f'{label} (Ground Truth)', marker='s')
            
            # Draw line connecting ground truth to predicted
            axs[1, 0].plot([true_x, pred_x], [true_y, pred_y], c=color, linestyle='--', alpha=0.7)

        axs[1, 0].set_title('PCA Coordinates: Predicted vs Ground Truth', fontsize=18)
        axs[1, 0].set_xlim(0, 1)
        axs[1, 0].set_ylim(0, 1)
        axs[1, 0].legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
        axs[1, 0].tick_params(axis='both', which='major', labelsize=12)
        axs[1, 0].set_aspect('equal')

        # Plot 4: Decoded 3D Landmarks
        ax = fig.add_subplot(224, projection='3d')
        all_landmarks = np.zeros((17, 3))
        colors_map = {}
        for i, (color, label) in enumerate(zip(colors, labels)):
            part_embedding = predicted_embeddings[i*2:i*2+2]
            decoded_landmarks = self.encoders[label].decode(part_embedding)
            if isinstance(decoded_landmarks, torch.Tensor):
                decoded_landmarks = decoded_landmarks.cpu().numpy()
            
            if label == 'upper':
                indices = self.upper_body_indices
            elif label == 'lower':
                indices = self.lower_body_indices
            elif label == 'l_arm':
                indices = self.l_arm_body_indices
            elif label == 'r_arm':
                indices = self.r_arm_body_indices
            
            all_landmarks[indices] = decoded_landmarks
            for idx in indices:
                colors_map[idx] = color

        # Plot 3D landmarks
        for i in range(17):
            if i in colors_map:
                x, z, y = all_landmarks[i]  # Swapped Y and Z
                ax.scatter(x, y, z, c=colors_map[i], s=100)
                ax.text(x, y, z, str(i), fontsize=8)

        # Define the correct connections for each body part
        upper_connections = [(4, 5), (4, 6), (5, 6)]
        lower_connections = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]
        l_arm_connections = [(5, 7), (7, 9), (5, 11)]
        r_arm_connections = [(6, 8), (8, 10), (6, 12)]

        # Draw lines connecting landmarks in 3D
        for connections, label in zip([upper_connections, lower_connections, l_arm_connections, r_arm_connections],
                                    ['upper', 'lower', 'l_arm', 'r_arm']):
            color = colors[labels.index(label)]
            for start, end in connections:
                if start in colors_map and end in colors_map:
                    ax.plot([all_landmarks[start, 0], all_landmarks[end, 0]],
                            [all_landmarks[start, 2], all_landmarks[end, 2]],  # Swapped Y and Z
                            [all_landmarks[start, 1], all_landmarks[end, 1]],  # Swapped Y and Z
                            c=color, linewidth=1, alpha=0.7)

        ax.set_title('Decoded 3D Landmarks', fontsize=18)
        ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, label=l, markersize=10) 
                        for c, l in zip(colors, labels)], fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')  # Swapped Y and Z
        ax.set_zlabel('Y')  # Swapped Y and Z

        # Ensure the aspect ratio is equal
        max_range = np.array([all_landmarks[:, 0].max() - all_landmarks[:, 0].min(),
                            all_landmarks[:, 2].max() - all_landmarks[:, 2].min(),  # Swapped Y and Z
                            all_landmarks[:, 1].max() - all_landmarks[:, 1].min()]).max() / 2.0  # Swapped Y and Z
        mid_x = (all_landmarks[:, 0].max() + all_landmarks[:, 0].min()) * 0.5
        mid_y = (all_landmarks[:, 2].max() + all_landmarks[:, 2].min()) * 0.5  # Swapped Y and Z
        mid_z = (all_landmarks[:, 1].max() + all_landmarks[:, 1].min()) * 0.5  # Swapped Y and Z
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Set the correct aspect ratio
        ax.set_box_aspect((1, 1, 1))

        # Adjust the view angle for better visibility
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = plt.imread(buf)
        plt.close(fig)
        
        img = np.transpose(img, (2, 0, 1))
        self.writer.add_image('Pose Analysis', img, step)

    def compute_regularization_loss(self, dataset, inputs):
        reg_batches = dataset.return_regularisation_batch(inputs)
        reg_losses = {}
        
        for part, batch in reg_batches.items():
            if part == 'upper':
                dim_start, dim_end = 0, 1
            elif part == 'lower':
                dim_start, dim_end = 2, 3
            elif part == 'l_arm':
                dim_start, dim_end = 4, 5
            elif part == 'r_arm':
                dim_start, dim_end = 6, 7
            else:
                raise ValueError(f"Unknown body part: {part}")

            inputs = batch['skeleton_2d'].view(batch['skeleton_2d'].size(0), -1).to(self.device)
        
            batch_embeddings = batch['embeddings']
            targets = torch.cat([
                batch_embeddings['upper'],
                batch_embeddings['lower'],
                batch_embeddings['l_arm'],
                batch_embeddings['r_arm']
            ], dim=1).float().to(self.device)
        
            outputs = self.model(inputs)
            variance = outputs[:, dim_start:dim_end].var(dim=0).mean()
            # loss = self.criterion(outputs, targets)

            
            reg_losses[part] = variance
        
        return reg_losses

    def train(self, train_dataset, val_dataset, batch_size=128, num_epochs=100, patience=10):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Collect all embeddings and skeletons for nearest neighbor search
        print("Compute knn")

        # Check if files already exist
        os.makedirs("./cached_knn_pca", exist_ok=True)
        embeddings_file = './cached_knn_pca/all_embeddings.pt'
        skeletons_file = './cached_knn_pca/all_skeletons.pt'

        if os.path.exists(embeddings_file) and os.path.exists(skeletons_file):
            print("Loading existing embeddings and skeletons...")
            all_embeddings = torch.load(embeddings_file).to(self.device)
            all_skeletons = torch.load(skeletons_file).to(self.device)
        else:
            print("Computing knn...")
            all_embeddings = []
            all_skeletons = []

            for batch in tqdm(train_loader, desc="Processing batches", unit="batch"):
                all_skeletons.append(batch[0]['skeleton_2d'])

                batch_embeddings = batch[0]['embeddings']
                targets = torch.cat([
                    batch_embeddings['upper'],
                    batch_embeddings['lower'],
                    batch_embeddings['l_arm'],
                    batch_embeddings['r_arm']
                ], dim=1).float().to(self.device)
                all_embeddings.append(targets)

            all_embeddings = torch.cat(all_embeddings, dim=0).to(self.device)
            all_skeletons = torch.cat(all_skeletons, dim=0).to(self.device)

            torch.save(all_embeddings, embeddings_file)
            torch.save(all_skeletons, skeletons_file)
            print(f"Saved {embeddings_file} and {skeletons_file}")        
            print("Compute knn finished")

        self.knn.add(all_embeddings, all_skeletons)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        step = 0

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                inputs = batch[0]['skeleton_2d'].view(batch[0]['skeleton_2d'].size(0), -1).to(self.device)
                batch_embeddings = batch[0]['embeddings']
                targets = torch.cat([
                    batch_embeddings['upper'],
                    batch_embeddings['lower'],
                    batch_embeddings['l_arm'],
                    batch_embeddings['r_arm']
                ], dim=1).float().to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)

                # reg_losses = self.compute_regularization_loss(train_dataset, batch[0])
                # reg_loss = sum(reg_losses.values())
                reg_loss = torch.zeros(1)
                # total_loss = loss + self.reg_weight * reg_loss
                total_loss = loss

                total_loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

                progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'reg_loss': f'{reg_loss.item():.4f}'})
                
                self.writer.add_scalar('Loss/BatchTrain', loss.item(), step)
                self.writer.add_scalar('Loss/RegLoss', reg_loss.item(), step)

                if step % 50 == 0:  # Visualize every 50 batches
                    nearest_skeleton, nearest_embedding = self.knn.find_nearest(outputs[0])
                    self.plot_and_compare_skeletons(
                        batch[0]['skeleton_2d'][0].detach().cpu().numpy(),
                        nearest_skeleton[0].detach().cpu().numpy(),
                        outputs[0].detach().cpu().numpy(), targets[0],
                        step
                    )
                    print(loss.item())

                step += 1

            self.model.eval()
            val_loss = 0.0
            val_reg_loss = {part: 0.0 for part in ['upper', 'lower', 'l_arm', 'r_arm']}
            
            print("Computing validation...")            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    inputs = batch[0]['skeleton_2d'].view(batch[0]['skeleton_2d'].size(0), -1).to(self.device)
                    batch_embeddings = batch[0]['embeddings']
                    targets = torch.cat([
                        batch_embeddings['upper'],
                        batch_embeddings['lower'],
                        batch_embeddings['l_arm'],
                        batch_embeddings['r_arm']
                    ], dim=1).float().to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()

                    reg_losses = self.compute_regularization_loss(val_dataset, batch[0])
                    for part, rl in reg_losses.items():
                        val_reg_loss[part] += rl.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            val_reg_loss = {part: loss / len(val_loader) for part, loss in val_reg_loss.items()}

            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            for part, loss in val_reg_loss.items():
                self.writer.add_scalar(f'RegLoss/Validation/{part}', loss, epoch)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val Reg Loss: {val_reg_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.save_model(f'poseregressor_pca_runs/{self.modelname}/best_')
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print("Early stopping")
                    break
        
        self.load_model(f'{self.modelname}_best_model')
        self.writer.close()
     
    def predict(self, skeleton_2d):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.FloatTensor(skeleton_2d).view(1, -1).to(self.device)
            outputs = self.model(inputs)
        return outputs.cpu().numpy()
    
    def save_model(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.model.state_dict(), f'{filename}_model.pth')
        with open(f'{filename}_encoders.pkl', 'wb') as f:
            pickle.dump(self.encoders, f)
    
    def load_model(self, filename):
        if filename.endswith('.pth'):
            model_file = filename
            encoders_file = filename.replace('.pth', '_encoders.pkl')
        else:
            model_file = f'{filename}_model.pth'
            encoders_file = f'{filename}_encoders.pkl'

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        if not os.path.exists(encoders_file):
            raise FileNotFoundError(f"Encoders file not found: {encoders_file}. "
                                    f"Expected to find it alongside the model file.")

        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        with open(encoders_file, 'rb') as f:
            self.encoders = pickle.load(f)

# Example usage:
if __name__ == "__main__":
    # Create datasets
    train_dataset = PosesDatasetPCA(use_additional_augment=False, split="train", fit=True)
    val_dataset = PosesDatasetPCA(use_additional_augment=False, split="test", fit=False)
    val_dataset.encoders = train_dataset.encoders

    # Create and train the model
    model = PoseRegressionModel(encoders=train_dataset.encoders, input_dim=34, output_dim=8)
    model.train(train_dataset, val_dataset)

    # Make predictions
    sample = val_dataset[0]
    skeleton_2d = sample['skeleton_2d'].numpy()
    predicted_embeddings = model.predict(skeleton_2d)
    print("Predicted embeddings shape:", predicted_embeddings.shape)

    # Save the model
    model.save_model('best_model.pth')

    # Load the model
    loaded_model = PoseRegressionModel(encoders=None, input_dim=34, output_dim=8)
    loaded_model.load_model('best_model.pth')