# Diffusion_examples

Training and inferring diffusion model on handwritten numbers, flowers, and food images using TensorFlow. 

Code taken (and slightly augmented to enable saving and importing simple "numpy-array image" datasets) 
from 
- https://keras.io/examples/generative/ddpm/  (description)
- https://github.com/keras-team/keras-io/blob/master/examples/generative/ddpm.py  (source code script)

written by A_K_Nain under the Apache License Version 2.0.
Diffusion of images by learning denoising using denoising diffusion probabilistic models, DDPMs.

#### Install requirements
```
python -m pip install -r requirements.txt
````

#### Download datasets
```
python download_datasets.py
````

#### Train & infer model
```
python diffusion.py
````

#### Miscellaneous
If using WSL(2) and missing ptxas, one can install cuda-nvcc in conda:
```bash
conda install -c nvidia cuda-nvcc 
```



![flowerGIF](.github/images/flower_diffusion.gif)

![foodGIF](.github/images/food_diffusion.gif)