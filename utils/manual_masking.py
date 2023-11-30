import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='')
parser.add_argument('--manual', action='store_true')
parser.add_argument('--mask', type=str, default='mask.png')
parser.add_argument('--output', type=str, default='output.png')

args = parser.parse_args()
image_path = args.input
mask = 0

if(args.manual):
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.title("Click on 4 points")
    points = plt.ginput(4, timeout=0)
    plt.close()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    x_coords, y_coords = zip(*points)
    plt.imshow(mask,cmap='gray')
    plt.axis('off')
    plt.fill(x_coords, y_coords, color='white')
    plt.savefig(args.mask, bbox_inches='tight', pad_inches=0)
    mask = cv2.imread(args.mask)

else:
    mask = cv2.imread('center_mask.png')

input = cv2.imread(image_path)
input_changed = cv2.resize(input,(256,256))
mask = cv2.resize(mask,(256,256))
mask = mask/255
mask_c = 1-mask
input_changed = input_changed * mask_c
input_changed = input_changed + mask*255
cv2.imwrite(args.output ,input_changed)
