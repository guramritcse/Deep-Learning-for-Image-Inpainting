import numpy as np
import torch
import torchvision.transforms as transforms
import os, cv2
import torchvision.utils as vutils
from argparse import ArgumentParser

# Set random seed for reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(42)
torch.manual_seed(40)
torch.cuda.manual_seed_all(40)

# Set CUDA device
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Argument parser
parser = ArgumentParser()
parser.add_argument('--mask', type=str, default='')
parser.add_argument('--input', type=str, default='')
parser.add_argument('--output', type=str, default='')
args = parser.parse_args()

# Define device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

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

"""
Test the model on test set
"""
# Load the last model 
model = torch.load('autoencoder.pth', map_location=torch.device(device))

# Test the model
with torch.no_grad():
    data = cv2.imread(args.input)
    data = cv2.resize(data, (128, 128))
    data_inp = torch.tensor(data).to(device)
    # Check if mask is provided, if not generate the mask
    if args.mask:
        data = data_inp
        data = data.cpu().permute(2,0,1).float()
        mask = get_mask()
        mask = torch.tensor(mask).to(device)
    else:
        data,mask = make_input(data_inp)
    # Get the output
    recon = model(data,mask)
    # Save the output
    vutils.save_image(recon, args.output, padding=0, normalize=True)
    