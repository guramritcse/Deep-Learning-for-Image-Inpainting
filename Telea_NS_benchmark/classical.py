# Inpainting using classical methods
import numpy as np
import cv2
import os
import torch.nn as nn
import sys
sys.path.append('..')
from utils.losses import mse, l1, psnr, lpips, tv

# Inpainting using Navier-Stokes algorithm
# https://docs.opencv.org/3.4/df/d3d/tutorial_py_inpainting.html
# https://docs.opencv.org/3.4/d7/d8b/group__photo__inpaint.html#gaedd30dfa0214fec4c88138b51d678085

"""
Get the masked image and the mask
"""
def get_masked(orig_img, img_sz=(256, 256)):
    img = orig_img.copy()
    img[img_sz[0]//4:3*img_sz[0]//4, img_sz[1]//4:3*img_sz[1]//4, :] = 0
    mask = np.zeros(img_sz, dtype=np.uint8)
    mask[img_sz[0]//4:3*img_sz[0]//4, img_sz[1]//4:3*img_sz[1]//4] = 255
    return img, mask


"""
Get the scores for the inpainted images
"""
def get_scores(dataset_path, save_path, img_sz, loss_fns=[nn.MSELoss()], method="telea", save=False):
    # extentions of the images
    extentions = ["png", "jpg", "jpeg", "bmp"]

    # list of scores
    scores = []

    # counter for number of images
    counter = 0

    # list of original and inpainted images
    orig_imgs = []
    out_imgs = []

    # create the save path if it doesn't exist
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    for entry in os.listdir(dataset_path):

        # Full path to the image
        full = os.path.join(dataset_path, entry)

        # Name of the image
        last = entry.split(".")[0]
        ext = entry.split(".")[-1]

        # Check if the file is an image
        if ext not in extentions:
            continue
        
        try:
            # Original image
            orig_img = cv2.imread(full)

            # Resize the image
            orig_img = cv2.resize(orig_img, img_sz)
            orig_imgs.append(orig_img)
        except:
            print("Error in resizing image: ", full)
            continue
        
        if save:
            cv2.imwrite(save_path + "/" + last + ".png", orig_img)
        
        # Masked image and mask
        img, mask = get_masked(orig_img, img_sz)
        if save:
            cv2.imwrite(save_path + "/" + last + "_input.png", img)
            cv2.imwrite(save_path + "/" + last + "_mask.png", mask)

        # Inpaint using Navier-Stokes algorithm
        if method == "ns":
            out = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
            out_imgs.append(out)
            if save:
                cv2.imwrite(save_path + "/" + last + "_ns.png", out)
        # Inpaint using Telea's algorithm
        else:
            out = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
            out_imgs.append(out)
            if save:
                cv2.imwrite(save_path + "/" + last + "_telea.png", out)

        counter += 1
        if counter % 100 == 0:
            print("Done with ", counter, " images for method: ", method)

    # Calculate the score
    for loss_fn in loss_fns:
        score = loss_fn(orig_imgs, out_imgs, img_sz)
        scores.append(score)

    return scores


if __name__ == "__main__":
    # path to dataset
    dataset_path = "../animals_test"
    # path to save the inpainted images
    save_path = "./animals_test_out"

    # get scores
    scores_telea = get_scores(dataset_path, save_path, (256, 256), loss_fns=[mse, l1, psnr, lpips, tv], method="telea")
    scores_ns = get_scores(dataset_path, save_path, (256, 256), loss_fns=[mse, l1, psnr, lpips, tv], method="ns")

    # get the average score
    print("Average MSE score for Telea's algorithm: ", scores_telea[0])
    print("Average MSE score for Navier-Stokes algorithm: ", scores_ns[0])
    print("Average L1 score for Telea's algorithm: ", scores_telea[1])
    print("Average L1 score for Navier-Stokes algorithm: ", scores_ns[1])
    print("Average PSNR score for Telea's algorithm: ", scores_telea[2])
    print("Average PSNR score for Navier-Stokes algorithm: ", scores_ns[2])
    print("Average LPIPS score for Telea's algorithm: ", scores_telea[3])
    print("Average LPIPS score for Navier-Stokes algorithm: ", scores_ns[3])
    print("Average TV score for Telea's algorithm: ", scores_telea[4])
    print("Average TV score for Navier-Stokes algorithm: ", scores_ns[4])
