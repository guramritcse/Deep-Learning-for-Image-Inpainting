# Deep Learning for Image Inpainting
This repository contains the code and report of the course project - Deep Learning for Image Inpainting, for the course CS 337 - Machine Learning and Artificial Intelligence, Autumn 2023, IIT Bombay.

### Team Members
| Name | Roll Number |
| --- | --- |
| Biradar Nikhil | 210050035 |
| Guramrit Singh | 210050061 |
| Omm Agrawal | 210050110 |
| Sabyasachi Samantaray | 210050138 |

## Abstract
Recent deep learning approaches to Image Inpainting have taken over the statistical methods which solely were
based on searching for identical patches in the valid regions.
These methods fail to produce features which cannot be found
in the rest of the image, essentially positive hallucination and
understanding the context of the image cannot be accounted
by statistical methods. Deep Learning Approaches have shown
promising results, yet they require large datasets. In this study
we replicate two such architectures proposed in literature, the
GLCIC and Contextual Attention based model and give a
comprehensive analytical review on a smaller dataset.

## Dataset
We have used [two datasets](https://iitbacin-my.sharepoint.com/:f:/g/personal/210050035_iitb_ac_in/EpTqLEiJSblNidfRT_pambQBmEcCwSiStBzHGn8w4HnGzw?e=IfDT4A) for our experiments:
* **Animals** - 5400 images of 90 different animals, with a train-test split of 4950-450
* **ImageNet** - 50000 images of 1000 different classes

## Models
The trained models can be found [here](https://iitbacin-my.sharepoint.com/:f:/g/personal/210050035_iitb_ac_in/EgWxZCttLY5PpnalGrCBiYIBTTu-RueRN5Xi34y9u6MBJA?e=UnRFiT)
### Baseline
1. **Navier-Stokes method** and **Fast Marching
Method** - These are the two statistical methods we have used as our baseline. The code for these methods can be found in the [Telea_NS_benchmark](./Telea_NS_benchmark/) folder.
2. **Autoencoder** - We have used a simple autoencoder as our baseline. The code for this can be found in the [Autoencoder](./Autoencoder/) folder.
### Deep Learning Models
1. **GLCIC** - The code for this can be found in the [GLCIC](./GLCIC/) folder.
2. **Contextual Attention** - The code for this can be found in the [Contextual_Attention](./Contextual-Attention/) folder.

## Requirements
We have used Python 3.9.2 for our experiments. The requirements can be found in the [requirements.txt](./requirements.txt) file. To install the requirements, run the following command:
```
pip install -r requirements.txt
```
For torch and torchvision installation, run the following command:
```
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

## Running the code
Before running the code, make sure that the dataset is downloaded and the path to the dataset is set correctly in the required file in the respective folders.
### Navier-Stokes method and Fast Marching Method 
* Navigate to [Telea_NS_benchmark](./Telea_NS_benchmark/) folder. 
* To run the code, run the following command:
```
python3 classical.py
```
* To get the inpainted image for a particular image, run the following command:
```
python3 test_single.py --input <path_to_image> --output <path_to_output_image> --mask <path_to_mask_image> --method <method_name (ns or telea)>
```

## Autoencoder
* Navigate to [Autoencoder](./Autoencoder/) folder. 
* For continuing training from a checkpoint or testing, download the `autoencoder.pth` file from [AutoEncoder](https://iitbacin-my.sharepoint.com/:f:/g/personal/210050035_iitb_ac_in/EgWxZCttLY5PpnalGrCBiYIBTTu-RueRN5Xi34y9u6MBJA?e=UnRFiT) and place it in the [Autoencoder](./Autoencoder/) folder.
* To train the model from scratch, run the following command:
```
python3 train.py --niter <number_of_iterations>
```
* To train the model from a checkpoint, run the following command:
```
python3 train.py --niter <number_of_iterations> --resume
```
* To get the inpainted image for a particular image, run the following command:
```
python3 test_single.py --input <path_to_image> --output <path_to_output_image> --mask <path_to_mask_image>
```
* To get the inpainted image for all the images in a folder, run the following command:
```
python3 test_all.py
```

## GLCIC
* Navigate to [GLCIC](./GLCIC/) folder.
* For continuing training from a checkpoint or testing, download the `animals/` folder from [GLCIC](https://iitbacin-my.sharepoint.com/:f:/g/personal/210050035_iitb_ac_in/EgWxZCttLY5PpnalGrCBiYIBTTu-RueRN5Xi34y9u6MBJA?e=UnRFiT) and place it in the [checkpoints](./GLCIC/checkpoints) folder.
* Configurations for training and testing can be found in [train.yaml](./GLCIC/configs/train.yaml) and [test.yaml](./GLCIC/configs/test.yaml) respectively inside the [configs](./GLCIC/configs/) folder and can be changed accordingly.
* To train the model, run the following command:
```
python3 train.py --config <path_to_config_file>
```
* To get the inpainted image for a particular image, run the following command:
```
python3 test_single.py --config <path_to_config_file> --input <path_to_image> --output <path_to_output_image> --mask <path_to_mask_image> --test_iter <iteration_number>
```
* To get the inpainted image for all the images in a folder, run the following command:
```
python3 test_all.py --config <path_to_config_file> --test_iter <iteration_number>
```

## Contextual Attention
* Navigate to [Contextual_Attention](./Contextual-Attention/) folder.
* For continuing training from a checkpoint or testing, download the `animals/` folder and/or `imagenet/` from [Contextual-Attention](https://iitbacin-my.sharepoint.com/:f:/g/personal/210050035_iitb_ac_in/EgWxZCttLY5PpnalGrCBiYIBTTu-RueRN5Xi34y9u6MBJA?e=UnRFiT) and place it in the [checkpoints](./Contextual-Attention/) folder.
* Configurations for training and testing can be found in [train.yaml](./Contextual-Attention/configs/train.yaml) and [test.yaml](./Contextual-Attention/configs/test.yaml) respectively inside the [configs](./Contextual-Attention/configs/) folder and can be changed accordingly.
* To train the model, run the following command:
```
python3 train.py --config <path_to_config_file>
```
* To get the inpainted image for a particular image, run the following command:
```
python3 test_single.py --config <path_to_config_file> --input <path_to_image> --output <path_to_output_image> --mask <path_to_mask_image> --test_iter <iteration_number>
```
* To get the inpainted image for all the images in a folder, run the following command:
```
python3 test_all.py --config <path_to_config_file> --test_iter <iteration_number>
```

## Utilities
The auxiliary functions used for training and testing can be found in the [utils](./utils/) folder.

### Manual Masking
To generate your own masked images, run the following command:
```
python3 manual_masking.py --manual --input <path_to_image> --output <path_to_output_image> --mask <path_to_output_mask>
```

## Report
The report, presentation and related material can be found in the [Report](./Report) folder.

## References 

[1] Yu, Jiahui & Lin, Zhe & Yang, Jimei & Shen, Xiaohui & Lu, Xin. (2018). Generative Image Inpainting with Contextual Attention. 5505-5514. 10.1109/CVPR.2018.00577. 

[2] Official Github, by authors of Contextual Attention: https://github.com/JiahuiYu/generative_inpainting/tree/v1.0.0, Reference codes by independent developers: [Daa223's Github Code](https://github.com/daa233/generative-inpainting-pytorch), [Zuroke's WaterMark Removal](https://github.com/zuruoke/watermark-removal), 
[WonwoongCho's Github Code](https://github.com/WonwoongCho/Generative-Inpainting-pytorch).
