
This code refer to [H2O](https://gc.com/) and [OmniH2O](https://omni.human2humanoid.com/):
- [Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation](https://gc.com/), IROS 2024.
- [OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning](https://omni.gc.com/), CoRL 2024.




# Motion Retargeting


## AMASS Dataset Preparation
Download [AMASS Dataset](https://amass.is.tue.mpg.de/index.html) with `SMPL + H G` format and put it under `gc/data/AMASS/AMASS_Complete/`:
```
|-- gc
   |-- data
      |-- AMASS
         |-- AMASS_Complete 
               |-- ACCAD.tar.bz2
               |-- BMLhandball.tar.bz2
               |-- BMLmovi.tar.bz2
               |-- BMLrub.tar
               |-- CMU.tar.bz2
               |-- ...
               |-- Transitions.tar.bz2

```

And then `cd gc/data/AMASS/AMASS_Complete` extract all the motion files by running:
```
for file in *.tar.bz2; do
    tar -xvjf "$file"
done
```

Then you should have:
```
|-- gc
   |-- data
      |-- AMASS
         |-- AMASS_Complete 
               |-- ACCAD
               |-- BioMotionLab_NTroje
               |-- BMLhandball
               |-- BMLmovi
               |-- CMU
               |-- ...
               |-- Transitions

```

## SMPL Model Preparation

Download [SMPL](https://smpl.is.tue.mpg.de/download.php) with `pkl` format and put it under `gc/data/smpl/`, and you should have:
```
|-- demo_retarget
   |-- data
      |-- smpl
         |-- SMPL_python_v.1.1.0.zip
```

Then `cd gc/data/smpl` and  `unzip SMPL_python_v.1.1.0.zip`, you should have 
```
|-- gc
   |-- data
      |-- smpl
         |-- SMPL_python_v.1.1.0
            |-- models
               |-- basicmodel_f_lbs_10_207_0_v1.1.0.pkl
               |-- basicmodel_m_lbs_10_207_0_v1.1.0.pkl
               |-- basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
            |-- smpl_webuser
            |-- ...
```
Rename these three pkl files and move it under smpl like this:
```
|-- gc
   |-- data
      |-- smpl
         |-- SMPL_FEMALE.pkl
         |-- SMPL_MALE.pkl
         |-- SMPL_NEUTRAL.pkl
```

## Retargeting AMASS to specific humanoid robot

We use an 3-step process to retarget the AMASS dataset to specific humanoid embodiments. Taking `H1` as an example here
1. Write forward kinematics of `H1` in `gc/phc/phc/utils/torch_h1_humanoid_batch.py`
2. Fit the SMPL shape that matches the `H1` kinematics in `gc/scripts/data_process/grad_fit_h1_shape.py`
3. Retarget the AMASS dataset based on the corresponding keypoints between fitted SMLP shape and `H1` using `gc/scripts/data_process/grad_fit_h1.py`

```
cd gc
python scripts/data_process/grad_fit_h1_shape.py
```

And you should have 
```
|-- gc
   |-- data
      |-- h1
         |-- shape_optimized_v1.pkl 
```

### Retargetting
   

```
cd gc
python scripts/data_process/grad_fit_h1.py
```
You should have:
```
(h2o) tairanhe@tairanhe-PRO-WS-WRX80E-SAGE-SE:~/Workspace/gc$ python scripts/data_process/grad_fit_h1.py
Importing module 'gym_38' (/home/tairanhe/Workspace/isaacgym/isaacgym/python/isaacgym/_bindings/linux-x86_64/gym_38.so)
Setting GYM_USD_PLUG_INFO_PATH to /home/tairanhe/Workspace/isaacgym/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugInfo.json
2024-07-11 18:35:43,587 - INFO - logger - logger initialized
  0%|                                                                                                                                                                                                                              | 0/15886 [00:00<?, ?it/s]15886 Motions to process
0-AMASS_Complete_MPI_Limits_03101_ulr1b_poses Iter: 0    256.983:   0%|                                                                                                                                                             | 0/15886 [00:01<?, ?it/s
```
After this retargeting loop done, you should have your embodiment-specific dataset ready.

To visualize the retargeted motion, you can run:

```
python scripts/vis/vis_motion.py
```


### Downloading Full retargeted motion dataset after feasibility filter: 
Download motion file `amass_phc_filtered.pkl` [here](https://cmu.box.com/s/vfi619ox7lwf2hzzi710p3g2l59aeczv), and put it under `gc/legged_gym/resources/motions/h1/amass_phc_filtered.pkl`. Make sure your running command overwrites the default motion file by `motion.motion_file=resources/motions/h1/amass_phc_filtered.pkl`


