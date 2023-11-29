# Deep Learning for Image Inpainting
This repository contains the code and report of the course project - Deep Learning for Image Inpainting, for the course CS 337 - Machine Learning and Artificial Intelligence, Autumn 2023, IIT Bombay.
> The report and presentation can be found in the `Report` folder.

### Team Lingua Franca
* Biradar Nikhil
* Guramrit Singh
* Omm Agrawal
* Sabyasachi Samantaray

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
comprehensive analytical review on a smaller dataset (5400
images of 90 different animals)

## Dataset
We have used [two datasets](https://iitbacin-my.sharepoint.com/:f:/g/personal/210050035_iitb_ac_in/EpTqLEiJSblNidfRT_pambQBmEcCwSiStBzHGn8w4HnGzw?e=IfDT4A) for our experiments:
* **Animals** - 5400 images of 90 different animals, with a train-test split of 4950-450
* **ImageNet** - 50000 images of 1000 different classes

## Models
The trained models can be found [here](https://iitbacin-my.sharepoint.com/:f:/g/personal/210050035_iitb_ac_in/EgWxZCttLY5PpnalGrCBiYIBTTu-RueRN5Xi34y9u6MBJA?e=UnRFiT)
### Baseline
1. **Navier-Stokes method** and **Fast Marching
Method** - These are the two statistical methods we have used as our baseline. The code for these methods can be found in the `Telea_NS_benchmark` folder.
2. **Autoencoder** - We have used a simple autoencoder as our baseline. The code for this can be found in the `Autoencoder` folder.
### Deep Learning Models
1. **GLCIC** - The code for this can be found in the `GLCIC` folder.
2. **Contextual Attention** - The code for this can be found in the `Contextual_Attention` folder.

## Requirements
We have used Python 3.9.2 for our experiments. The requirements can be found in the `requirements.txt` file. To install the requirements, run the following command:
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
* Navigate to `Telea_NS_benchmark` folder. 
* To run the code, run the following command:
```
python3 classical.py
```
* To get the inpainted image for a particular image, run the following command:
```
python3 test_single.py --input <path_to_image> --output <path_to_output_image> --mask <path_to_mask_image> --method <method_name (ns or telea)>
```

## Autoencoder
* Navigate to `Autoencoder` folder. 
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
* Navigate to `GLCIC` folder.
* Configurations for training and testing can be found in `train.yaml` and `test.yaml` respectively inside the `configs` folder.
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
* Navigate to `Contextual_Attention` folder.
* Configurations for training and testing can be found in `train.yaml` and `test.yaml` respectively inside the `configs` folder.
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

## References
[1] Pathak, Deepak & Krahenbuhl, Philipp & Donahue, Jeff & Darrell, Trevor & Efros, Alexei. (2016). Context Encoders: Feature Learning by Inpainting. 2536-2544. 10.1109/CVPR.2016.278.

[2] Iizuka, Satoshi & Simo-Serra, Edgar & Ishikawa, Hiroshi. (2017). Globally and locally consistent image completion. ACM Transactions on Graphics. 36. 1-14. 10.1145/3072959.3073659. 

[3] Yu, Jiahui & Lin, Zhe & Yang, Jimei & Shen, Xiaohui & Lu, Xin. (2018). Generative Image Inpainting with Contextual Attention. 5505-5514. 10.1109/CVPR.2018.00577. 

[4] Official Github, by authors of Contextual Attention: https://github.com/JiahuiYu/generative_inpainting/tree/v1.0.0, Reference codes by independent developers: [Daa223's Github Code](https://github.com/daa233/generative-inpainting-pytorch), [Zuroke's WaterMark Removal](https://github.com/zuruoke/watermark-removal), 
[WonwoongCho's Github Code](https://github.com/WonwoongCho/Generative-Inpainting-pytorch).
