import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from regression_dataset_pca_2inputs import PosesDatasetPCA
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
    def __init__(self, input_dim=68, output_dim=8, input_projection_dim=1024):
        super(PoseRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_projection_dim),
            nn.BatchNorm1d(input_projection_dim),
            nn.ReLU(),
            ResidualBlock(input_projection_dim, 256),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            ResidualBlock(256, 256),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResidualBlock(128, 128),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
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
    def __init__(self, encoders, input_dim=68, output_dim=51, learning_rate=0.0005):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.encoders = encoders
        self.model = PoseRegressor(input_dim, output_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.modelname = f"PCA2IN_{current_time};"
        self.writer = SummaryWriter(f'pose_3d_regressor_skeleton_runs/{self.modelname}')
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


    def plot_and_compare_skeletons(self, skeleton, skeleton2, skeleton3d, predicted_3d, step):
        fig, axs = plt.subplots(2, 2, figsize=(20, 20), dpi=100)
        fig.suptitle('Pose Analysis', fontsize=24)

        # Helper function to add index labels
        def add_index_labels(ax, points):
            for i, (x, y) in enumerate(points):
                ax.text(x, y, str(i), fontsize=10, ha='center', va='center', 
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        # Plot 1: First 2D Skeleton projection
        axs[0, 0].scatter(skeleton[:, 0], skeleton[:, 1], c='b', s=50)
        for start, end in BONES_COCO:
            axs[0, 0].plot([skeleton[start, 0], skeleton[end, 0]],
                        [skeleton[start, 1], skeleton[end, 1]], 'r-', linewidth=2)
        add_index_labels(axs[0, 0], skeleton)
        axs[0, 0].set_title('First 2D Skeleton projection', fontsize=18)
        axs[0, 0].set_aspect('equal')
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])

        # Plot 2: Second 2D Skeleton projection
        axs[0, 1].scatter(skeleton2[:, 0], skeleton2[:, 1], c='b', s=50)
        for start, end in BONES_COCO:
            axs[0, 1].plot([skeleton2[start, 0], skeleton2[end, 0]],
                        [skeleton2[start, 1], skeleton2[end, 1]], 'r-', linewidth=2)
        add_index_labels(axs[0, 1], skeleton2)
        axs[0, 1].set_title('Second 2D Skeleton projection', fontsize=18)
        axs[0, 1].set_aspect('equal')
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])

        # Helper function to plot 3D skeleton
        def plot_3d_skeleton(ax, skeleton, title):
            ax.scatter(skeleton[:, 0], skeleton[:, 2], skeleton[:, 1], c='b', s=50)  # Swapped Y and Z
            for i, (x, y, z) in enumerate(skeleton):
                ax.text(x, z, y, str(i), fontsize=8)  # Swapped Y and Z
            for start, end in BONES_COCO:
                ax.plot([skeleton[start, 0], skeleton[end, 0]],
                        [skeleton[start, 2], skeleton[end, 2]],  # Swapped Y and Z
                        [skeleton[start, 1], skeleton[end, 1]],  # Swapped Y and Z
                        c='r', linewidth=1, alpha=0.7)
            ax.set_title(title, fontsize=18)
            ax.set_xlabel('X')
            ax.set_ylabel('Z')  # Swapped Y and Z
            ax.set_zlabel('Y')  # Swapped Y and Z
            
            # Ensure the aspect ratio is equal
            max_range = np.array([skeleton[:, 0].max() - skeleton[:, 0].min(),
                                skeleton[:, 2].max() - skeleton[:, 2].min(),  # Swapped Y and Z
                                skeleton[:, 1].max() - skeleton[:, 1].min()]).max() / 2.0  # Swapped Y and Z
            mid_x = (skeleton[:, 0].max() + skeleton[:, 0].min()) * 0.5
            mid_y = (skeleton[:, 2].max() + skeleton[:, 2].min()) * 0.5  # Swapped Y and Z
            mid_z = (skeleton[:, 1].max() + skeleton[:, 1].min()) * 0.5  # Swapped Y and Z
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=20, azim=45)

        # Plot 3: Ground Truth 3D Skeleton
        ax_gt = fig.add_subplot(223, projection='3d')
        plot_3d_skeleton(ax_gt, skeleton3d, 'Ground Truth 3D Skeleton')

        # Plot 4: Predicted 3D Skeleton
        ax_pred = fig.add_subplot(224, projection='3d')
        plot_3d_skeleton(ax_pred, predicted_3d, 'Predicted 3D Skeleton')

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

    def train(self, train_dataset, val_dataset, batch_size=128, num_epochs=300, patience=50):
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

        upper_indices = slice(0, 2)
        lower_indices = slice(2, 4)
        l_arm_indices = slice(4, 6)
        r_arm_indices = slice(6, 8)

        def flatten_poses(poses):
            """
            Flatten a batch of 3D poses from [batch_size, 17, 3] to [batch_size, 51]
            """
            return poses.reshape(poses.size(0), -1)

        def unflatten_poses(flat_poses):
            """
            Reshape a batch of flattened poses from [batch_size, 51] to [batch_size, 17, 3]
            """
            return flat_poses.reshape(flat_poses.size(0), 17, 3)

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                inputs = batch[0]['skeleton_2d'].view(batch[0]['skeleton_2d'].size(0), -1).to(self.device)
                inputs2 = batch[0]['skeleton_2d_2'].view(batch[0]['skeleton_2d'].size(0), -1).to(self.device)

                inputs = torch.cat([inputs, inputs2], dim = 1)

                batch_embeddings = batch[0]['embeddings']
                # targets = torch.cat([
                #     batch_embeddings['upper'],
                #     batch_embeddings['lower'],
                #     batch_embeddings['l_arm'],
                #     batch_embeddings['r_arm']
                # ], dim=1).float().to(self.device)

                target = flatten_poses(batch[0]['skeleton_3d']).to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.criterion(outputs, target)

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
                # self.writer.add_scalar('Loss/RegLoss', reg_loss.item(), step)

                # self.writer.add_scalar('Loss/UpperBody', loss_upper.item(), step)
                # self.writer.add_scalar('Loss/LowerBody', loss_lower.item(), step)
                # self.writer.add_scalar('Loss/LeftArm', loss_l_arm.item(), step)
                # self.writer.add_scalar('Loss/RightArm', loss_r_arm.item(), step)

                if step % 50 == 0:  # Visualize every 50 batches
                    self.plot_and_compare_skeletons(
                        batch[0]['skeleton_2d'][0].detach().cpu().numpy(),
                        batch[0]['skeleton_2d_2'][0].detach().cpu().numpy(),
                        batch[0]['skeleton_3d'][0].detach().cpu().numpy(),
                        unflatten_poses(outputs)[0].detach().cpu().numpy(),
                        # nearest_skeleton[0].detach().cpu().numpy(),
                        # outputs[0].detach().cpu().numpy(), targets[0],
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
                    inputs2 = batch[0]['skeleton_2d_2'].view(batch[0]['skeleton_2d'].size(0), -1).to(self.device)
                    inputs = torch.cat([inputs, inputs2], dim = 1)

                    # batch_embeddings = batch[0]['embeddings']

                    target = flatten_poses(batch[0]['skeleton_3d']).to(self.device)
                    # targets = torch.cat([
                    #     batch_embeddings['upper'],
                    #     batch_embeddings['lower'],
                    #     batch_embeddings['l_arm'],
                    #     batch_embeddings['r_arm']
                    # ], dim=1).float().to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, target)
                    val_loss += loss.item()

                    # reg_losses = self.compute_regularization_loss(val_dataset, batch[0])
                    # for part, rl in reg_losses.items():
                    #     val_reg_loss[part] += rl.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            # val_reg_loss = {part: loss / len(val_loader) for part, loss in val_reg_loss.items()}

            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            # for part, loss in val_reg_loss.items():
            #     self.writer.add_scalar(f'RegLoss/Validation/{part}', loss, epoch)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            # print(f"Val Reg Loss: {val_reg_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.save_model(f'pose_3d_regressor_skeleton_runs/{self.modelname}/best_.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print("Early stopping")
                    break
        
        self.save_model(f'pose_3d_regressor_skeleton_runs/{self.modelname}/_best_model.pth')
        self.writer.close()
     
    def predict(self, skeleton_2d):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.FloatTensor(skeleton_2d).view(1, -1).to(self.device)
            outputs = self.model(inputs)
        return outputs.cpu().numpy()
    
    def save_model(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.model.state_dict(), filename)
    
    def load_model(self, model_filename):
        # if filename.endswith('.pth'):
        #     model_file = filename
        #     encoders_file = filename.replace('.pth', '_encoders.pkl')
        # else:
        #     model_file = f'{filename}_model.pth'
        #     encoders_file = f'{filename}_encoders.pkl'

        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"Model file not found: {model_filename}")

        self.model.load_state_dict(torch.load(model_filename, map_location=self.device))

# Example usage:
if __name__ == "__main__":
    # Create datasets
    train_dataset = PosesDatasetPCA(use_additional_augment=False, split="train", fit=False)
    val_dataset = PosesDatasetPCA(use_additional_augment=False, split="test", fit=False)

    # Just loading these for every model for now
    with open("pca_encoder_final.pkl", 'rb') as f:
        train_dataset.encoders = pickle.load(f)
        val_dataset.encoders = train_dataset.encoders

    # Create and train the model
    model = PoseRegressionModel(encoders=train_dataset.encoders, input_dim=68, output_dim=51)
    model.train(train_dataset, val_dataset)

    # Make predictions
    sample = val_dataset[0]
    skeleton_2d = sample['skeleton_2d'].numpy()
    predicted_embeddings = model.predict(skeleton_2d)
    print("Predicted embeddings shape:", predicted_embeddings.shape)

    # Save the model
    model.save_model('pose_2d_regressor_2inputs_skeleton_best_model.pth')

    # Load the model
    loaded_model = PoseRegressionModel(encoders=None, input_dim=34, output_dim=8)
    loaded_model.load_model('best_model.pth', "best_model_encoders.pkl")