# PointCNN using Pytorch
**Dataset: ModelNet40**

## Reference
This work is based on [hxdengBerkeley's PointCNN.Pytorch](https://github.com/hxdengBerkeley/PointCNN.Pytorch). A Pytorch implementation of PointCNN.

## Improvements
Many improvements have been done to increase the GPU efficiency (was 95% of the time at 0% on GPU but CPU was consistantly busy) and significantly speed up the training process. Most of the calculations in model have been move to GPU but they can still be switched back to CPU. When training on ModelNet40 Dataset, a minimum of 3.5Gb GPU MEM are expected. While there are some CPU compulsory calculations, the overall GPU efficiency is around 85%. 

* Move a majority of computational workload to GPU from CPU
* Rewrite the self-defined data loader to `torch.utils.data.Dataset` and `torch.utils.data.Dataloader`
* Add ```adam``` as an alternative optimizer
* Add generating and saving plot of train/test accuracy and train loss
* _Adjustable train set size (randomly select data but considering distribution over classes)**(Under implementation)**_

## Environment
This project relies on a variety of packages. To make the environment building-up process more straightaway, a configuration file, ```xconv_conda_env_setup.yml```, is provided. It is recommended to use Anaconda to install the environment.

For getting and installing Anaconda, Please visit [Anaconda's official website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to download and install 
the Anaconda on your devices.

## Setting up ENV and Training
If you already have the ENV set up, simply jump over step 2.

1. ```cd``` to the project directory _(where ```xconv_conda_env_setup.yml``` is located at)_
2. In your terminal, use ```conda create  --name env_name --file xconv_conda_env_setup.yml```. Replace ```env_name``` to your desired environment name.
3. In your terminal, use ```conda activate env_name``` to switch to this environment.
4. Prepare data according to [yangyanli/PointCNN](https://github.com/yangyanli/PointCNN)
5. In your terminal, use ```python train_pytorch_modelnet40.py``` to start training. There are multiple commandline arguments acceptable. 
6. Have fun