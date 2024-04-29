# PiRN
## Reproduce Physics-Driven Turbulence Image Restoration with Stochastic Refinement. 
## Hua Tong*, Bruce Jia*, Yujie Zheng 

### Task Definition
According to the description of the [paper](https://arxiv.org/pdf/2307.10603.pdf), we use part of the original dataset to train the model. Then use the model to restore the images. Finally, we estimate the restored images with ground truth images and calculate the PSNR and SSIM.

## Getting started
### Prerequisites
1 . Clone this repository
```
git clone https://github.com/Hua78158452/PiRN-EC522-Final.git
```

2 . Prepare feature files and data
Download [FFHQ](https://app.box.com/s/njcngbxvfrhy476fdjpkwmfozo71twpu), which is the dataset author used. However, they did not tell us how to use the dataset. So we tried many times and changed the path in the train.py to make it work. We accidentally found that when the path is a certain one, the code can run.

Dataset used to inference 

Download[OTIS](https://zenodo.org/records/161439)
Download[CLEAR](https://uob-my.sharepoint.com/:f:/g/personal/eexna_bristol_ac_uk/EnEq5HdW_ThImbQmKNE8dBoBy3CvXy_yqE4023_GbSoJBQ?e=vMB6Xg)
Download[heat_chamber_new](https://drive.google.com/file/d/14iVachB95bCCtke8ONPD9CCH20JO75v2/view?usp=sharing)

After downloading the feature file, extract it to **YOUR DATA STORAGE** directory:

3 . Install dependencies.
- Python 
- PyTorch 
- Cuda 
- tensorboard
- tqdm
- lmdb
- easydict
- msgpack
- msgpack_numpy

To install the dependencies use conda and pip, we suggest using pip command to install any lost dependencies. Chatgpt can help if you do not know how to install them.
for example:
```
pip install easydict lmdb msgpack msgpack_numpy
```

### Training, Inference & Estimate 
*NOTE*: Currently only support train and inference using one GPU. 
1 .  PiRN training
For successful training, we change the batch size and configuration parameters many times, the details are found in the codes we upload.
For the train.py to run successfully in SCC(You will do the same thing when you run other codes), we need to create a virtual Python environment:
```
python3 -m venv myenv
source myenv/bin/activate
```
then activate training:
```
python3 train.py
```
2 .  PiRN inference
After training, you will find the mode, which is a .pth file, in the checkpoint folder, use the one that has the biggest number.
Before your inference, please check the model you used by checking the configurations.yaml, line 49.
Then make sure your input path contains the pictures, which are used to infer:
```
python3 inference.py
```
3 .  Estimate
After inference, you can find output images in inference_output folder. Please remember to manage them. Then as for the requirement of estimate_psnr.py, you need to put all your restored images's paths in the test_data_lq.txt, put all the ground_truth images' paths in the test_data_gt.txt, and you need to make sure they are matched. 
```
python3 estimate_psnr.py
```
Then we can get the PSNR and SSIM to estimate the model we trained before.

## Acknowledgement
We thank the authors for open-sourcing the great project!

## Contact
huatong@bu.edu
