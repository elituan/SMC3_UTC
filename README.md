This is the repository for the submission to [SMCDC2021: Smoky Mountains Computational Sciences and Engineering Conference Data Challenge - Challenge 3](https://smc-datachallenge.ornl.gov/2021-challenge-3/)
The code is developed based on the repository of [ Hierarchical Multi-Scale Attention for Semantic Segmentation](https://github.com/NVIDIA/semantic-segmentation)

## Installation 

* The code is tested with pytorch 1.3 and python 3.6
* You need to install apex to run the code:
```bash
  > cd apex-master
  > pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
````
If the script above does not work please try
```bash
  > cd apex-master
  > pip install -v --disable-pip-version-check --no-cache-dir ./
````
* Install requirement
```bash
  > pip install -r requirements.txt
````


## Run Inference

* Download pretrained weights from [google drive](https://drive.google.com/drive/folders/1zTaw8Wm5IIvOCIQLFr3hxvJLlWpUrLqB?usp=sharing) and save it to folder weight
```bash
  > mkdir <weight>
````
* Create a directory where you save the testing images. In this case, we save image from cityscapes and SoftRainNoon (CARLA)
```bash
  > mkdir <imgs>
  > mkdir <imgs/Cityscapes>
  > mkdir <imgs/SoftRainNoon>
```

* Run the script below to get inference image. Remember to change the <eval_folder> and <result_dir>. We should keep the image from each domain into a seperated folder
```bash
> python -m torch.distributed.launch --nproc_per_node=1 train.py --dataset smc --cv 0 --apex --fp16 --bs_val 1 --eval folder --eval_folder ./imgs/SoftRainNoon/ --dump_all_images --n_scales 0.5,1.0,2.0 --snapshot ./weight/best_checkpoint_ep33.pth --arch ocrnet.HRNet_Mscale --result_dir ./imgs/SoftRainNoon_prediction/ --datasetBN
> python -m torch.distributed.launch --nproc_per_node=1 train.py --dataset smc --cv 0 --apex --fp16 --bs_val 1 --eval folder --eval_folder ./imgs/Cityscapes/ --dump_all_images --n_scales 0.5,1.0,2.0 --snapshot ./weight/best_checkpoint_ep33.pth --arch ocrnet.HRNet_Mscale --result_dir ./imgs/Cityscapes_prediction/ --datasetBN
```
## Training
to be updated

