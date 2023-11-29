import os
import random
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from models import Generator
from ..utils.tools import get_config, random_bbox, mask_image, is_image_file, image_loader, normalize, get_model_list
from dataset import Dataset

# Parse arguments
parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/test.yaml')
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
print("Random seed: {}".format(seed))
random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed_all(seed)

"""
Test model on test set
"""
with torch.no_grad(): 
    # Define the dataset
    test_dataset = Dataset(data_path=config['test_in_dir'],
                image_shape=config['image_shape'],
                random_crop=config['random_crop'])
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=config['batch_size'],
                                            shuffle=False,
                                            num_workers=config['num_workers'])

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
    netG.load_state_dict(torch.load(last_model_name,  map_location=torch.device('cpu')))
    model_iteration = int(last_model_name[-11:-3])
    print(f"Loaded model from {checkpoint_path} at iteration {model_iteration}")

    iterable_test_loader = iter(test_loader)

    img_num = 0
    while True:
        try:
            ground_truth = next(iterable_test_loader)
        except StopIteration:
            break

        # Make mask and input
        bboxes = random_bbox(config, batch_size=ground_truth.size(0))
        data, mask = mask_image(ground_truth, bboxes, config)
        mask2 = torch.zeros_like(mask)
        
        if cuda:
            netG = nn.parallel.DataParallel(netG, device_ids=device_ids)
            data = data.cuda()
            mask = mask.cuda()
            mask2 = mask2.cuda()
            netG = netG.to("cuda")
            ground_truth = ground_truth.cuda()

        # Get the inpainted result
        _, patch_recon = netG(data, mask, mask2)
        recon = patch_recon * mask + data * (1. - mask)

        for i in range(ground_truth.shape[0]):
            # Save the inpainted result
            if config['test_img_save']:
                vutils.save_image(ground_truth[i],
                            f'{config["test_out_dir"]}/{img_num}_orig.png',
                            nrow=1 * 1,
                            normalize=True)
                vutils.save_image(data[i],
                            f'{config["test_out_dir"]}/{img_num}_mask.png',
                            nrow=1 * 1,
                            normalize=True)
                vutils.save_image(recon[i],
                            f'{config["test_out_dir"]}/{img_num}_recon.png',
                            nrow=1 * 1,
                            normalize=True)
                img_num+=1
