import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os, cv2
from models import AutoEncoder
from argparse import ArgumentParser

# Set random seed for reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(42)
torch.manual_seed(40)
torch.cuda.manual_seed_all(40)

# Set CUDA device
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Log file
log_file = open("log.txt", "w")

# Argument parser
parser = ArgumentParser()
parser.add_argument('--niter', type=int, default=1200)
parser.add_argument('--resume', type=bool, default=False)
args = parser.parse_args()

# Define device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
log_file.write(f"Device: {device}\n")

"""
Load the dataset
"""
def read_image_tensor(image_folder,transform):
    torch.cuda.empty_cache()
    all_images = []
    for images in os.listdir(image_folder):
        img = torchvision.io.read_image(os.path.join(image_folder,images)).float()
        all_images.append(transform(img))
    print(f"Done with Loading Dataset")
    return torch.stack(all_images).to(device)

"""
Generate the mask
"""
def get_mask():
    mask = np.zeros((128,128,3),dtype=np.float32)
    mask[32:96,32:96,:]=255
    return mask

"""
Generate the input image
"""
def make_input(labels):
    images = []
    mask = get_mask()
    mask[mask==255] = 1
    for label in labels:
        img = np.array(label.cpu().permute(1,2,0).int())
        img = cv2.resize(img, (128, 128))
        gen_img = img*(1-mask)
        gen_img[gen_img==0] = 255
        gen_img = np.array(torch.tensor(gen_img).cpu().permute(2,0,1).float())
        images.append(gen_img)
    mask[mask==1] = 255
    mask = np.array(torch.tensor(mask).cpu().permute(2,0,1).float())
    return torch.tensor(np.array(images)).to(device),torch.tensor(mask).to(device)

# default image size
img_size = (128,128)

# define the transforms
base_transform = transforms.Compose(
    [transforms.Resize(img_size)]
)

# read the dataset
train_set = read_image_tensor("../animals_train", base_transform)
test_set = read_image_tensor("../animals_test", base_transform)

# print the size of the dataset
log_file.write(f"Train set: {len(train_set)}\n")
log_file.write(f"Test set: {len(test_set)}\n")

# define the dataloader
train_loader = DataLoader(dataset=train_set,
                        batch_size=32,
                        shuffle=True)
test_loader = DataLoader(dataset=test_set,
                        batch_size=32,
                        shuffle=False)

# if resume is true, load the last model
if args.resume:
    model = torch.load('autoencoder.pth', map_location=torch.device(device))
else:
    model = AutoEncoder(in_channels=3, out_channels=3)
model.to(device)

"""
Train the model
"""
def train_model(model, loss_fn, optimizer, num_epochs, train_loader):
    train_losses=[]
    for epoch in range(num_epochs):
        loss_batches=[]
        for data in train_loader:
            img = data
            img = img.to(device)
            optimizer.zero_grad()
            transformed_img,mask = make_input(img)
            transformed_img = transformed_img.float()
            mask = mask.float()
            output= model(transformed_img,mask)
            loss = loss_fn(output, img)
            loss_batches.append(loss.item())
            loss.backward()
            optimizer.step()
        torch.save(model, f'autoencoder.pth')
        avg_train_loss=sum(loss_batches)/len(loss_batches)
        train_losses.append(avg_train_loss)
        log_file.write('Epoch [{}/{}], Loss: {:.4f}\n'.format(epoch+1, num_epochs, avg_train_loss))
        print('Epoch [{}/{}], Loss: {:.4f}\n'.format(epoch+1, num_epochs, avg_train_loss))
    return train_losses

# Loss function and optimizer
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Number of epochs
epochs = args.niter

# Train the model
train_losses=train_model(model, loss, optimizer, epochs, train_loader)
log_file.write(f'train_loss: {train_losses}\n')


"""
Test the model on test set
"""
# Load the last model 
model = torch.load('autoencoder.pth', map_location=torch.device(device))

# Test the model
with torch.no_grad():
    losses = []
    for data in test_loader:
        data = data.to(device)
        data, mask = make_input(data)
        recon = model(data,mask)
        losses.append(nn.MSELoss()(recon, data).item())
    log_file.write(f"Average Test Reconstruction Loss: {sum(losses)/len(losses)}\n")
