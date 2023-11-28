import cv2
import argparse

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--mask', type=str, default='')
parser.add_argument('--input', type=str, default='')
parser.add_argument('--output', type=str, default='')
parser.add_argument('--method', type=str, default='telea')
args = parser.parse_args()

# Open the image.
img = cv2.imread(args.input)
img = cv2.resize(img, (256, 256))

# Load the mask.
mask = cv2.imread(args.mask, 0)
mask = cv2.resize(mask, (256, 256))

# Inpaint.
if args.method == 'telea':
    dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
else:
    dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

# Write the output.
cv2.imwrite(args.output, dst)
