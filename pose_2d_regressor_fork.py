import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from regression_dataset_poses_cached import PosesDataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import io
import os
from tqdm import tqdm
from datetime import datetime

BONES_COCO = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    (0, 5), (0, 6)
]


# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.bn = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU(inplace=False)  # Changed to non-inplace ReLU

#     def forward(self, x):
#         return self.relu(self.bn(self.conv(x)))

# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = ConvBlock(channels, channels)
#         self.conv2 = ConvBlock(channels, channels)
#         self.relu = nn.ReLU(inplace=False)  # Changed to non-inplace ReLU

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = out + residual  # Changed from += to +
#         return self.relu(out)

# class EnhancedPoseRegressor(nn.Module):
#     def __init__(self, input_dim=34, output_dim=8, num_residual_blocks=3):
#         super(EnhancedPoseRegressor, self).__init__()
        
#         self.input_proj = nn.Linear(input_dim, 64)
        
#         self.conv_blocks = nn.Sequential(
#             ConvBlock(2, 32),
#             ConvBlock(32, 64),
#             ConvBlock(64, 128)
#         )
        
#         self.residual_blocks = nn.Sequential(
#             *[ResidualBlock(128) for _ in range(num_residual_blocks)]
#         )
        
#         self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
#         self.fc_layers = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(inplace=False),  # Changed to non-inplace ReLU
#             nn.Linear(64, output_dim)
#         )

#     def forward(self, x):
#         batch_size = x.size(0)
#         x = self.input_proj(x)
#         x = x.view(batch_size, 2, -1)  # Reshape to (batch_size, 2, 32) for 32 joints
        
#         x = self.conv_blocks(x)
#         x = self.residual_blocks(x)
        
#         x = self.global_avg_pool(x).squeeze(-1)
#         x = self.fc_layers(x)
        
#         return x

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
    def __init__(self, input_dim=34, output_dim=8):
        super(PoseRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResidualBlock(128, 128),
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

    def __init__(self, input_dim=34, output_dim=8, learning_rate=0.0015): # originally 0.001
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model = PoseRegressor(input_dim, output_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.writer = SummaryWriter()
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.writer = SummaryWriter(f'fork_runs/FORK_{current_time}')
        self.knn = FastKNN()

    def plot_and_compare_skeletons(self, skeleton, nearest_skeleton, step):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        skeletons = [skeleton, nearest_skeleton]
        titles = ['Original 2D Skeleton', 'Nearest Neighbor 2D Skeleton']
        
        for ax, skel, title in zip([ax1, ax2], skeletons, titles):
            ax.scatter(skel[:, 0], skel[:, 1], c='b', s=20)
            for start, end in BONES_COCO:
                ax.plot([skel[start, 0], skel[end, 0]],
                        [skel[start, 1], skel[end, 1]], 'r-')
            ax.set_title(title)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=200)
        buf.seek(0)
        img = plt.imread(buf)
        plt.close(fig)
        
        img = np.transpose(img, (2, 0, 1))
        self.writer.add_image('Skeletons Comparison', img, step)

    def normalize_umap_targets(self, targets):
        return (targets + 7) / 37

    def denormalize_umap_predictions(self, preds):
        return preds * 37 - 7

    def train(self, train_dataset, val_dataset, batch_size=128, num_epochs=100, patience=10):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)


        # Collect all embeddings and skeletons for nearest neighbor search
        print("Compute knn")

        # Check if files already exist
        os.makedirs("./cached_knn", exist_ok=True)
        embeddings_file = './cached_knn/all_embeddings.pt'
        skeletons_file = './cached_knn/all_skeletons.pt'

        if os.path.exists(embeddings_file) and os.path.exists(skeletons_file):
            print("Loading existing embeddings and skeletons...")
            all_embeddings = torch.load(embeddings_file).to(self.device)
            all_skeletons = torch.load(skeletons_file).to(self.device)
        else:
            print("Computing knn...")
            all_embeddings = []
            all_skeletons = []

            # Wrap train_loader with tqdm for progress bar
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

            # Save the tensors to data_knn folder
            torch.save(all_embeddings, embeddings_file)
            torch.save(all_skeletons, skeletons_file)
            print(f"Saved {embeddings_file} and {skeletons_file}")


        self.knn.add(all_embeddings, all_skeletons)
        
        print("Compute knn finished")
       

       

        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        step = 0

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                inputs = batch[0]['skeleton_2d'].view(batch[0]['skeleton_2d'].size(0), -1).to(self.device)
        
                batch_embeddings = batch[0]['embeddings']
                targets = torch.cat([
                    self.normalize_umap_targets(batch_embeddings['upper']),
                    self.normalize_umap_targets(batch_embeddings['lower']),
                    self.normalize_umap_targets(batch_embeddings['l_arm']),
                    self.normalize_umap_targets(batch_embeddings['r_arm'])
                ], dim=1).float().to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                self.writer.add_scalar('Loss/BatchTrain', loss.item(), step)


                # Find nearest neighbor and visualize
                if step % 50 == 0:  # Visualize every 100 batches
                    nearest_skeleton, nearest_embedding = self.knn.find_nearest(self.denormalize_umap_predictions(outputs[0]))
                    self.plot_and_compare_skeletons(batch[0]['skeleton_2d'][0].detach().cpu().numpy(), nearest_skeleton[0].detach().cpu().numpy(), step)
  
                    print(loss.item())

                step += 1

                train_loss += loss.item()
            

            self.model.eval()
            
            
            print("Computing validation...")
            
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0]['skeleton_2d'].view(batch[0]['skeleton_2d'].size(0), -1).to(self.device)
            
                    batch_embeddings = batch[0]['embeddings']
                    targets = torch.cat([
                        self.normalize_umap_targets(batch_embeddings['upper']),
                        self.normalize_umap_targets(batch_embeddings['lower']),
                        self.normalize_umap_targets(batch_embeddings['l_arm']),
                        self.normalize_umap_targets(batch_embeddings['r_arm'])
                    ], dim=1).float().to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            self.writer.add_scalar('Loss/EpochTrain', train_loss, step)
            self.writer.add_scalar('Loss/Validation', val_loss, step)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), 'fork_best_model.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print("Early stopping")
                    break
        
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.writer.close()
     
    def predict(self, skeleton_2d):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.FloatTensor(skeleton_2d).view(1, -1).to(self.device)
            outputs = self.model(inputs)
        return outputs.cpu().numpy()
    
    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)
    
    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))

# Example usage:
if __name__ == "__main__":
    # Create datasets
    train_dataset = PosesDataset(use_additional_augment=False, split="train")
    val_dataset = PosesDataset(use_additional_augment=False, split="test")

    # Create and train the model
    model = PoseRegressionModel(input_dim=34, output_dim=8)
    model.train(train_dataset, val_dataset)

    # Make predictions
    sample = val_dataset[0]
    skeleton_2d = sample['skeleton_2d'].numpy()
    predicted_embeddings = model.predict(skeleton_2d)
    print("Predicted embeddings shape:", predicted_embeddings.shape)

    # Save the model
    model.save_model('best_model.pth')

    # Load the model
    loaded_model = PoseRegressionModel(input_dim=34, output_dim=8)
    loaded_model.load_model('best_model.pth')