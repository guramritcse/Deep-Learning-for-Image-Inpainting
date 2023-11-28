import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import TotalVariation

"""
All the loss functions are defined in here

Any loss function has the following signature:
Input:
  orig_imgs: list of original images (python list of numpy arrays)
  out_imgs: list of inpainted images (python list of numpy arrays)
  img_sz: size of the images (tuple of integers)
Output:
  score: score of the inpainted images (float)

The following loss functions are defined in this file:
  mse: Mean Squared Error
  l1: Mean Absolute Error
  psnr: Peak Signal to Noise Ratio
  fid: Fréchet Inception Distance
  lpips: Learned Perceptual Image Patch Similarity
  tv: Total Variation
"""

# Set seed for reproducibility
def set_seed():
    torch.manual_seed(123)
    np.random.seed(42)

# Mean Squared Error
def mse(orig_imgs, out_imgs, img_sz=(256, 256)):
    set_seed()
    scores = []
    for orig_img, out in zip(orig_imgs, out_imgs):
        mse = nn.MSELoss()(torch.from_numpy(out).float(), torch.from_numpy(orig_img).float())
        scores.append(mse.item())
    return np.mean(scores)

# Mean Absolute Error
def l1(orig_imgs, out_imgs, img_sz=(256, 256)):
    set_seed()
    scores = []
    for orig_img, out in zip(orig_imgs, out_imgs):
        l1 = nn.L1Loss()(torch.from_numpy(out).float(), torch.from_numpy(orig_img).float())
        scores.append(l1.item())
    return np.mean(scores)

# Peak Signal to Noise Ratio
def psnr(orig_imgs, out_imgs, img_sz=(256, 256)):
    set_seed()
    scores = []
    for orig_img, out in zip(orig_imgs, out_imgs):
        mse = nn.MSELoss()(torch.from_numpy(out).float(), torch.from_numpy(orig_img).float())
        if mse == 0:
            print("MSE is 0")
            scores.append(100)
        # max pixel value
        PIXEL_MAX = np.max(orig_img)
        scores.append(20 * np.log10(PIXEL_MAX / np.sqrt(mse)))
    return np.mean(scores)

# Resize an array of images to a new size
def resize_images(images, new_shape):
    set_seed()
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)

# scale images to [0,1]
def scale_images(images):
    set_seed()
    images_list = list()
    for image in images:
        new_image = image.astype('float32')
        new_image = new_image / 255.0
        images_list.append(new_image)
    return np.asarray(images_list)

# The FID (Fréchet Inception Distance) score is commonly calculated using the InceptionV3 model, 
# The choice of the InceptionV3 model is based on its effectiveness in capturing features relevant to 
# image quality and diversity. 
# The FID score compares the statistics of feature representations (specifically, the activations in one of the intermediate layers) 
# of real and generated images.
# Reference: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/#:~:text=The%20Frechet%20Inception%20Distance%20score,for%20real%20and%20generated%20images.
def fid(orig_imgs, out_imgs, img_sz=(256, 256)):
    set_seed()
    # prepare the inception v3 model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    # resize images
    images1 = resize_images(orig_imgs, (299,299,3))
    images2 = resize_images(out_imgs, (299,299,3))
    # pre-process images
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

## LPIPS - Learned Perceptual Image Patch Similarity
# Reference: https://torchmetrics.readthedocs.io/en/stable/image/learned_perceptual_image_patch_similarity.html
# A low LPIPS score means that image patches are perceptual similar.
# Both input image patches are expected to have shape (N, 3, H, W). 
# The minimum size of H, W depends on the chosen backbone (see net_type arg).
def lpips(orig_imgs, out_imgs, img_sz=(256, 256)):
    set_seed()
    images1 = resize_images(orig_imgs, (3, 128, 128))
    images2 = resize_images(out_imgs, (3, 128, 128))
    images1 = scale_images(images1)
    images2 = scale_images(images2)
    images1 = torch.from_numpy(images1).float()
    images2 = torch.from_numpy(images2).float()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', reduction='mean', normalize=True)
    return lpips(images1, images2).item()

def tv(orig_imgs, out_imgs, img_sz=(256, 256)):
    set_seed()
    tv = TotalVariation(reduction="mean")
    images1 = np.asarray(orig_imgs)
    images1 = torch.from_numpy(images1).float()
    images2 = np.asarray(out_imgs)
    images2 = torch.from_numpy(images2).float()
    score = (tv(images1) - tv(images2)) / (img_sz[0]*img_sz[1])
    return score.item()
