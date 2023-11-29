import os
import random
import time
import shutil
from argparse import ArgumentParser
import gc
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from trainer import Trainer
from GLCIC.dataset import Dataset
from ..utils.tools import get_config, random_bbox, mask_image, get_logger

# Parse arguments
parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/train.yaml')
parser.add_argument('--seed', type=int, help='manual seed')

torch.autograd.set_detect_anomaly(True)

def main():
    gc.enable()
    torch.cuda.empty_cache()
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

    # Define the checkpoint path
    checkpoint_path = os.path.join('checkpoints',
                                   config['dataset_name'],
                                   config['mask_type'] + '_' + config['expname'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    shutil.copy(args.config, os.path.join(checkpoint_path, os.path.basename(args.config)))
    writer = SummaryWriter(logdir=checkpoint_path)
    logger = get_logger(checkpoint_path)  

    logger.info("Arguments: {}".format(args))
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    logger.info("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)


    logger.info("Configuration: {}".format(config))

    try:
        # Define the dataset
        logger.info("Training on dataset: {}".format(config['dataset_name']))
        train_dataset = Dataset(data_path=config['train_data_path'],
                                image_shape=config['image_shape'],
                                random_crop=config['random_crop'])
        val_dataset = Dataset(data_path=config['val_data_path'],
                              image_shape=config['image_shape'],
                              random_crop=config['random_crop'])
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=config['batch_size'],
                                                   shuffle=True,
                                                   num_workers=config['num_workers'])
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                  batch_size=config['batch_size'],
                                                  shuffle=False,
                                                  num_workers=config['num_workers'])
        
        # Define the trainer
        trainer = Trainer(config)
        logger.info("\n{}".format(trainer.netG))
        logger.info("\n{}".format(trainer.localD))
        logger.info("\n{}".format(trainer.globalD))

        if cuda:
            trainer = nn.parallel.DataParallel(trainer, device_ids=device_ids)
            trainer_module = trainer.module
        else:
            trainer_module = trainer

        # Get the resume iteration to restart training
        start_iteration = trainer_module.resume(checkpoint_path,config['resume']) if config['resume'] else 1

        iterable_train_loader = iter(train_loader)
        iterable_val_loader = iter(val_loader)

        time_count = time.time()

        for iteration in range(start_iteration, config['niter'] + 1):
            iterable_train_loader = iter(train_loader)
            total_losses = {}
            while(1):
                torch.cuda.empty_cache()
                gc.collect()

                try:
                    ground_truth = next(iterable_train_loader)
                except StopIteration:
                    break

                # Prepare the inputs
                bboxes = random_bbox(config, batch_size=ground_truth.size(0))
                x, mask = mask_image(ground_truth, bboxes, config)

                # Create mask2 same size as mask and all 0
                mask2 = torch.zeros_like(mask)
                if cuda:
                    x = x.cuda()
                    mask = mask.cuda()
                    ground_truth = ground_truth.cuda()
                    mask2 = mask2.cuda()

                # Forward pass
                compute_g_loss = iteration % config['n_critic'] == 0
                losses, inpainted_result = trainer(x, bboxes, mask, mask2, ground_truth, compute_g_loss)

                for k in losses.keys():
                    if not losses[k].dim() == 0:
                        losses[k] = torch.mean(losses[k])
                

                # Backward pass
                # Update D
                trainer_module.optimizer_d.zero_grad()
                losses['d'] = losses['wgan_d'] + losses['wgan_gp'] * config['wgan_gp_lambda']
                losses['d'].backward(retain_graph=True)
                # Update G
                if compute_g_loss:
                    trainer_module.optimizer_g.zero_grad()
                    losses['g'] = losses['l1'] * config['l1_loss_alpha'] + losses['ae'] * config['ae_loss_alpha'] + losses['wgan_g'] * config['gan_loss_alpha']
                    losses['g'].backward()
                    trainer_module.optimizer_g.step()

                trainer_module.optimizer_d.step()

                for k in losses.keys():
                    if not k in total_losses:
                        total_losses[k] = [losses[k].item()]
                    else:
                        total_losses[k].append(losses[k].item())       

            for k in total_losses.keys():
                total_losses[k] = torch.mean(torch.tensor(total_losses[k]))

            log_losses = ['l1', 'ae', 'wgan_g', 'wgan_d', 'wgan_gp', 'g', 'd']
            if iteration % config['print_iter'] == 0:
                time_count = time.time() - time_count
                speed = config['print_iter'] / time_count
                speed_msg = 'speed: %.2f batches/s ' % speed
                time_count = time.time()

                message = 'Iter: [%d/%d] ' % (iteration, config['niter'])
                for k in log_losses:
                    v = total_losses.get(k, 0.)
                    writer.add_scalar(k, v, iteration)
                    message += '%s: %.6f ' % (k, v)
                message += speed_msg
                logger.info(message)

            if iteration % (config['viz_iter']) == 0:
                viz_max_out = config['viz_max_out']
                try:
                    val_ground_truth = next(iterable_val_loader)
                except StopIteration:
                    iterable_val_loader = iter(val_loader)
                    val_ground_truth = next(iterable_val_loader)

                # Prepare the inputs
                val_bboxes = random_bbox(config, batch_size=val_ground_truth.size(0))
                val_x, val_mask = mask_image(val_ground_truth, val_bboxes, config)
                # Create mask2 same size as mask and all 0
                val_mask2 = torch.zeros_like(val_mask)
                if cuda:
                    val_x = val_x.cuda()
                    val_mask = val_mask.cuda()
                    val_ground_truth = val_ground_truth.cuda()
                    val_mask2 = val_mask2.cuda()
                compute_g_loss = iteration % config['n_critic'] == 0
                val_losses, val_inpainted_result = trainer(val_x, val_bboxes, val_mask, val_mask2, val_ground_truth, compute_g_loss)

                if val_x.size(0) > viz_max_out:
                    viz_images = torch.stack([val_x[:viz_max_out], val_inpainted_result[:viz_max_out]], dim=1)
                else:
                    viz_images = torch.stack([val_x, val_inpainted_result], dim=1)
                viz_images = viz_images.view(-1, *list(val_x.size())[1:])
                vutils.save_image(viz_images,
                                '%s/niter_%03d.png' % (checkpoint_path, iteration),
                                nrow=2 * 4,
                                normalize=True)

            # Save the model
            if iteration % config['snapshot_save_iter'] == 0:
                trainer_module.save_model(checkpoint_path, iteration)
            
            del total_losses
        

    except Exception as e:
        logger.error("{}".format(e))
        raise e


if __name__ == '__main__':
    s_time = time.time()
    main()
    print("##########################################")
    print(f"Total Time = {time.time()-s_time}")
