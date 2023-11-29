import os
import random
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from models import Generator
from ..utils.tools import get_config, image_loader, normalize, get_model_list

# Parse arguments
parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/test.yaml')
parser.add_argument('--input', type=str, default='')
parser.add_argument('--mask', type=str, default='')
parser.add_argument('--output', type=str, default='output.png')
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--test_iter', type=int, default=0)
args = parser.parse_args()

# Load experiment setting
config = get_config(args.config)

# Define device and configure CUDA
cuda = config['cuda']
device_ids = config['gpu_ids']
if cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
    device_ids = list(range(len(device_ids)))
    config['gpu_ids'] = device_ids
    cudnn.benchmark = True

# Set random seed for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed_all(seed)

"""
Test model on test set
"""
with torch.no_grad():             
    data = image_loader(args.input)
    data = transforms.Resize(config['image_shape'][:-1])(data)
    data = transforms.CenterCrop(config['image_shape'][:-1])(data)
    data = transforms.ToTensor()(data)
    data = normalize(data)

    mask = image_loader(args.mask)
    mask = transforms.Resize(config['image_shape'][:-1])(mask)
    mask = transforms.CenterCrop(config['image_shape'][:-1])(mask)
    mask = transforms.ToTensor()(mask)[0].unsqueeze(dim=0)

    mask2 = torch.zeros_like(mask)

    data = data * (1. - mask)
    data = data.unsqueeze(dim=0)
    mask = mask.unsqueeze(dim=0)

    # Define the checkpoint path
    if not args.checkpoint_path:
        checkpoint_path = os.path.join('checkpoints',
                                        config['dataset_name'],
                                        config['mask_type'] + '_' + config['expname'])
    else:
        checkpoint_path = args.checkpoint_path

    # Define the model
    netG = Generator(config['netG'], cuda, device_ids)

    # Load the trained model
    last_model_name = get_model_list(checkpoint_path, "gen", iteration=args.test_iter)
    netG.load_state_dict(torch.load(last_model_name))
    model_iteration = int(last_model_name[-11:-3])
    print(f"Loaded model from {checkpoint_path} at iteration {model_iteration}")

    if cuda:
        netG = nn.parallel.DataParallel(netG, device_ids=device_ids)
        data = data.cuda()
        mask = mask.cuda()
        mask2 = mask2.cuda()
        netG = netG.to("cuda")

    # Get the inpainted result
    _, patch_recon = netG(data, mask, mask2)
    recon = patch_recon * mask + data * (1. - mask)

    # Save the inpainted result
    vutils.save_image(recon, args.output, padding=0, normalize=True)
    print("Saved the inpainted result to {}".format(args.output))
