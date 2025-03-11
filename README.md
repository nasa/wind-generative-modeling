# Wind Diffusion Modeling

This repo contains the code and data used to reproduce the results in the paper:
```
T. A. Shah, M. C. Stanley, and J. E. Warner. Generative Modeling of Microweather Wind Velocities for Urban Air Mobility. IEEE Aerospace Conference. 2025.
```


![](images/2022-04-25.gif)

*An animation of a single day of sodar sensor readings (taken in 2 minute intervals) with their corresponding macroweather forecast.*

## Abstract

Motivated by the pursuit of safe, reliable, and weather-tolerant urban air mobility (UAM) solutions, this work proposes a generative modeling approach for characterizing microweather wind velocities. 
Microweather, or the weather conditions in highly localized areas, is particularly complex in urban environments owing to the chaotic and turbulent nature of wind flows. 
Furthermore, traditional means of assessing local wind fields are not generally viable solutions for UAM applications: 1) field measurements that would rely on permanent wind profiling systems in operational air space are not practical, 2) physics-based models that simulate fluid dynamics at a sufficiently high resolution are not computationally tractable, and 3) data-driven modeling approaches that are largely deterministic ignore the inherent variability in turbulent flows that dictates UAM reliability. 
Thus, advancements in predictive capabilities are needed to help mitigate the unique operational safety risks that microweather winds pose for smaller, lighter weight UAM aircraft. 

This work aims to model microweather wind velocities in a manner that is computationally-efficient, captures random variability, and would only require a temporary, rather than permanent, field measurement campaign. 
Inspired by recent breakthroughs in conditional generative AI such as text-to-image generation, the proposed approach learns a probabilistic macro-to-microweather mapping between regional weather forecasts and measured local wind velocities using generative modeling. 
A simple proof of concept was implemented using a dataset comprised of local (micro) measurements from a Sonic Detection and Ranging (SoDAR) wind profiler along with (macro) forecast data from a nearby weather station over the same time period. 
Generative modeling was implemented using both state of the art deep generative models (DGMs), denoising diffusion probabilistic models and flow matching, as well as the well-established Gaussian mixture model (GMM) as a simpler baseline. 
Using current macroweather forecasted wind speed and direction as input, the results show that the proposed macro-to-microweather conditional generative models can produce statistically consistent wind velocity vs. altitude samples, capturing the random variability in the localized measurement region. 
While the simpler GMM performs well for unconditional wind velocity sample generation, the DGMs show superior performance for conditional sampling and provide a more capable foundation for scaling to larger scale measurement campaigns with denser spatial/temporal sensor readings.

## Installation

Create an environment.
```
conda create -n wind python=3.10
```

Install packages from conda
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda install lightning -c conda-forge

conda install -c conda-forge notebook
```

Install packages from pip
```
pip install matplotlib pandas einops tqdm optuna seaborn gpytorch
```

Install local source
```
pip install -e .
```

<!-- ### Running on K Cluster GPU Nodes:

**Setting up pytorch virtual environment with Conda**

- module purge
- module load anaconda/3_2022.10
- conda create --name K_pytorch_env
- conda activate K_pytorch_env
- module load cuda/11.8.0
- conda install --yes --file requirements.txt

**Running (in interactive session)**

- qsub -I -q K5-res-A100-PL -l walltime=12:00:00 -lselect=1:ncpus=4:ngpus=4:mem=20G
- conda activate K_pytorch_env
- module load cuda/11.8.0
- python ...

For more info on using GPUs on the K Cluster, see “Running GPU Jobs” section on https://k-info.larc.nasa.gov/CCFhowto_jobsubmitt.html -->

## Training

In order to train the models run the scripts within the `scripts/training` folder.

For example:
```
python scripts/training/train_fm_unconditional.py
```

This will generate a checkpoint folder called `results` which contains all saved models.

## Sampling

Evaluation of each model's sampling capabilities can be done by running the `scripts/sampling` scripts.
First you must modify the path of the model.
For example, specify the model you with to sample from:

```
path = Path('results/fm_unconditional/devices-1_bs-128_lr-0.0001/version_0/checkpoints/epoch=399-step=20800.ckpt')
```

And then run the file:

```
python scripts/sampling/sample_fm_unconditional.py
```

Which will generate a `.csv` file containing the generated samples.
If a conditional model is used, the corresponding macroweather condition for each sample will be appended as a column.

## Postprocessing

Each `.py` file within the `scripts/processing` folder generates figures demonstrating each model's performance.
Call each script to generate figures from the paper.

## Model Notes:

- Current diffusion model implementation (with UNet architecture) is based on: https://huggingface.co/blog/annotated-diffusion
- Flow matching implementation based on: https://github.com/atong01/conditional-flow-matching

## Notices:

Copyright © 2025 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.
 
The Generative Modeling of Wind Velocity Measurements Software (LAR-20572-1) calls the following third-party software, which is subject to the terms and conditions of its licensor, as applicable at the time of licensing.  The third-party software is not bundled or included with this software but may be available from the licensor.  License hyperlinks are provided here for information purposes only:
 
PyTorch
BSD
https://github.com/pytorch/pytorch/blob/main/LICENSE
Copyright (c) 2016- Facebook, Inc  (Adam Paszke)
Copyright (c) 2014- Facebook, Inc (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006 diap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
 
Scikit-learn
BSD
https://github.com/scikit-learn/scikit-learn/blob/main/COPYING
Copyright (c) 2007-2024 The scikit-learn developers.
All rights reserved.
 
NumPy
BSD
https://numpy.org/devdocs/license.html
Copyright (c) 2005-2025, NumPy Developers.
All rights reserved.
 
SciPy
BSD
https://github.com/scipy/scipy/blob/main/LICENSE.txt
Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
All rights reserved.
 
Matplotlib
PSF
https://matplotlib.org/stable/project/license.html
 
seaborn
BSD
https://github.com/mwaskom/seaborn/blob/master/LICENSE.md
Copyright (c) 2012-2023, Michael L. Waskom All rights reserved.
 
## Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS." 
 
## Waiver and Indemnity:  
RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS, AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
