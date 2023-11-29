import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os, cv2

# Set random seed for reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(42)
torch.manual_seed(40)
torch.cuda.manual_seed_all(40)

# Set CUDA device
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Define device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

"""
Load the dataset
"""
def read_image_tensor(image_folder,transform):
    torch.cuda.empty_cache()
    all_images = []
    for images in os.listdir(image_folder):
        img = torchvision.io.read_image(os.path.join(image_folder,images)).float()
        all_images.append(transform(img))
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
data_path = "../animals_test"
dataset = read_image_tensor(data_path, base_transform)

# define the dataloader
test_loader = DataLoader(dataset=dataset,
                        batch_size=32,
                        shuffle=False)

"""
Test the model on test set
"""
# Load the last model
model = torch.load('autoencoder.pth', map_location=torch.device('cpu'))

# Make the output directory if it doesn't exist
save_path = "./animals_test_out"
if not os.path.exists(save_path):
    os.makedirs(save_path)

img_num = 0
with torch.no_grad():
    for data in test_loader:
        data_inp = data.to(device)
        data,mask = make_input(data_inp)
        recon = model(data,mask)
        # Save the images
        for i in range(len(data)):
            plt.imsave(f"./animals_test_out/{img_num}_orig.png",data_inp[i].int().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8))
            plt.imsave(f"./animals_test_out/{img_num}_mask.png",data[i].int().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8))
            plt.imsave(f"./animals_test_out/{img_num}_recon.png",recon[i].int().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8))
            img_num += 1
