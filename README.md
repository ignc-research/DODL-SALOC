# DODL-SALOC
The repository contains the test code for the paper: Odometryless Indoor Dynamic Object Detection and Localization with Spatial Anticipation

To test the medthod, you can download the chech point and data. Put them under the data folder. We recommand to use conda environment. We provide one example. 
More dataset can be aquired under https://github.com/ignc-research/habitatdyn. We are working on constructing the dataset and training environment, it will be opensource soon.
```
pip install -r requirements.txt
python eval/test_on_example.py
```
The two video will be crated when you choose the IF_IM parameter with True or False:

# The video with SALOC
https://github.com/ignc-research/DODL-SALOC/assets/68584274/41d51c94-e3bf-4d6f-b0eb-2d43f5d47708


# The video without SALOC
https://github.com/ignc-research/DODL-SALOC/assets/68584274/38d121aa-f7ae-4288-baff-0f5d28e4225c

