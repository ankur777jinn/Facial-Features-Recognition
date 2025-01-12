# %% [markdown]
# Importing the necessary Libraries for the Code:

# %%
import os
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# %% [markdown]
# Function for loading the Images from The datasets :

# %%
def load_metadata(meta_data_path, image_folder):
  
    pairs = []
    labels = []

    for relation_folder in os.listdir(image_folder):
        relation_path = os.path.join(image_folder, relation_folder)

        mat_file = os.path.join(meta_data_path, f"{relation_folder}.mat")
        data = loadmat(mat_file)['pairs']
        for pair in data:
                # Extract image filenames and construct full paths
                img1 = os.path.join(relation_path, str(pair[2][0]))
                img2 = os.path.join(relation_path, str(pair[3][0]))
                label = pair[1][0]  # 1 for similar, 0 for dissimilar
                
                pairs.append((img1, img2))
                labels.append(label)
                
    return pairs, labels

# %%
def preprocess_image(img_path):
    image=cv.imread(img_path)
    image_resized = cv.resize(image, (224, 224))
    image_normalized = image_resized / 255.0 
    image_transposed = np.transpose(image_normalized, (2, 0, 1)) #As pytorch prefers the image tensore in form of (Chnnel, Height, Width)
    return torch.tensor(image_transposed, dtype=torch.float32)

# %%
def loadmat_data(meta_data_path, image_folder):
    pairs, labels = load_metadata(meta_data_path, image_folder)
  
    preprocessed_pairs = []
    for img1_path, img2_path in pairs:

            img1 = preprocess_image(img1_path)
            img2 = preprocess_image(img2_path)
            preprocessed_pairs.append((img1, img2))

    labels_np = np.array(labels, dtype=np.float32) # Converting labels to numpy array first, then to a torch tensor
    return preprocessed_pairs, torch.tensor(labels_np, dtype=torch.float32)

# %%
meta_data_path = r"E:\SNN project\data\KinFaceW-II\meta_data"
image_folder = r"E:\SNN project\data\KinFaceW-II\images"

preprocessed_pairs, labels = loadmat_data(meta_data_path, image_folder)

print(f"Number of pairs loaded: {len(preprocessed_pairs)}")
print(f"Sample label: {labels[0]}")


pair_sample = preprocessed_pairs[0]
print(f"Shape of first image: {pair_sample[0].shape}")
print(f"Shape of second image: {pair_sample[1].shape}")

# %%
class SNN_ARCHITECTURE(nn.Module): # inheriting from base class nn.Module
  def __init__(self):   #initializer of this class
    super(SNN_ARCHITECTURE,self).__init__() #calling the initializer of the base class

    self.cnn = nn.Sequential(                                            # 2 convolutional layers
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.fc = nn.Sequential(                           # two fully connected layers
      nn.Linear(64 * 56 * 56, 512),
      nn.ReLU(),
      nn.Linear(512, 128)
    )

  def forward(self, x):
    x = self.cnn(x)
    x=torch.flatten(x,1)
    x = self.fc(x)
    return x
  
  def Euclidean_distance(self, image_1, image_2):
   
    embed_1=self.forward(image_1)
    embed_2=self.forward(image_2)
    euclidean_distance = F.pairwise_distance(embed_1, embed_2)

    return euclidean_distance

# %%
class Loss_Fxn(nn.Module):
    def __init__(self,margin=1.5):
        super(Loss_Fxn, self).__init__()
        self.margin=margin

    def forward(self, euclidean_distance, label):
        Contrastive_loss =0.5*label*(euclidean_distance)**2+(1-label)*torch.clamp(self.margin-euclidean_distance, min=0)**2
        return torch.mean(Contrastive_loss) #Contrastive_loss


# %%
class Inherit_Dataset(Dataset):
    def __init__(self, preprocessed_pairs, labels):
        self.pairs = preprocessed_pairs
        self.labels = labels
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1, img2 = self.pairs[idx]
        label = self.labels[idx]
        return img1, img2, label

# %%
# Create the Dataset and DataLoader
inherit_dataset = Inherit_Dataset(preprocessed_pairs, labels)  # Correct variable name
dataloader = DataLoader(inherit_dataset, batch_size=16, shuffle=True)  # Correct DataLoader instance

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SNN_ARCHITECTURE().to(device)
criterion = Loss_Fxn(margin=1.5)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# %%
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for img1, img2, label in dataloader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        
        # Forward pass
        distance = model.Euclidean_distance(img1, img2)

        # Compute loss
        loss = criterion(distance, label)
        epoch_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")


# %%
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    distance = model.Euclidean_distance(img1, img2)
    # Perform your inference logic here


# %%

def visualize_results(img1, img2, distance):
    img1 = img1.permute(1, 2, 0).cpu().numpy()
    img2 = img2.permute(1, 2, 0).cpu().numpy()
    
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title('Image 1')

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title('Image 2')

    plt.suptitle(f'Euclidean Distance: {distance.item():.4f}')
    plt.show()

# Ensure correct variable name for dataset instance
dataset = Inherit_Dataset(preprocessed_pairs, labels)

# Example usage
with torch.no_grad():
    x=np.random.randint(0, len(dataset))
    img1, img2, _ = dataset[x]
    distance = model.Euclidean_distance(img1.unsqueeze(0).to(device), img2.unsqueeze(0).to(device))
    visualize_results(img1, img2, distance)



# %%
def get_embeddings(model, dataloader):
    model.eval()  # Set model to evaluation mode
    embeddings = []
    labels = []
    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2 = img1.to(device), img2.to(device)
            embed_1 = model.forward(img1)
            embed_2 = model.forward(img2)
            embeddings.extend(embed_1.cpu().numpy())
            embeddings.extend(embed_2.cpu().numpy())
            labels.extend(label.cpu().numpy())
            labels.extend(label.cpu().numpy())
    return np.array(embeddings), np.array(labels)


# %%


def apply_tsne(embeddings):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)
    return tsne_embeddings


# %%
def plot_tsne(tsne_embeddings, labels):
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.7)
    legend1 = plt.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    plt.gca().add_artist(legend1)
    plt.title('t-SNE of Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()


# %%
# Get embeddings from the model
embeddings, labels = get_embeddings(model, dataloader)

# Apply t-SNE to the embeddings
tsne_embeddings = apply_tsne(embeddings)

# Plot the t-SNE results
plot_tsne(tsne_embeddings, labels)



