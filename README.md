# budE (boosting urban depth estimations)
A simple pipeline combining [IM2ELE's](https://github.com/speed8928/IMELE) depth estimation for aerial data and [BoostingMonocularDepth's](https://github.com/compphoto/BoostingMonocularDepth) monocular boosting to retrieve higher resolution depth estimations for urban scenes! `budE` can produce clearer building boundaries, demonstrates an increased tolerance to estimating crowded images with high hieght variability, and is designed to allow for integration of future (or existing) aerial or satellite depth estimation networks.

[![video](./video_thumbnail.jpg)](https://www.youtube.com/watch?v=rhLK7ONcKWI)

# Getting Started

## Model Weights
To run `budE` first clone the repository then download the required path files and put them in their proper locations.

- Download the IM2ELE SNET weights from [here](http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth) and put it in:
> budE/im2ele/models

- Next, download the IM2ELE weights from [here]() and put it in:
> budE/im2ele/

- Finally, download the BoostingMonocularDepths weights from [here]() and put it in:
> budE/boostingMonocularDepth/pix2pix/checkpoints/mergemodel/

## Data 
The test aerial data for the IM2ELE model can be found [here](https://drive.google.com/drive/folders/14sBkjeYY7R1S9NzWI5fGLX8XTuc8puHy?usp=sharing) in the `test_rgbs_base` directory. If you don't have your own dataset to boost, download that one to get started! Note that this is aerial data of Ireland released as part of the Ordinance Survey Ireland project.

## Boosting
Once you have the model weights downloaded and have ready-to-boost data, simply run

```{bash}
python boost_urban_aerial.py --bilateral_blur --data_dir test_rgbs_base/ --output_dir outputs/boosted_results 
```
If you do not want your boost results to be blurred with a bilateral filter (edge-preserving) simply omit the `--bilateral_blur` argument from the command above. Currently, `budE` uses a `wholeimage_size_threshold` of 700. This means that the boosting network will never resize the image to a resolution larger than 700. If you want to interactively set the `wholeimage_size_threshold` instead, run 

```{bash}
python boost_urban_aerial.py --bilateral_blur --interactive_max --data_dir test_rgbs_base/ --output_dir outputs/boosted_results 
```

 # Using Your Own Model
Do you have a CNN aerial or satellite depth estimation model you want to try out? Great news! `budE` is not exclusive to the IM2ELE depth estimation. To include another model for boosting through `budE` simply
- Include the model info in a seperate directory, e.g:
> budE/yourDepthEstModel/

`yourDepthEstModel/` should contain (1) the model definition/class (2) the model weights, and (3) a function which takes one (un-transformed) image as a NumPy array, uses your model to estimate a depth estimation ([see im2ele/estimate.py](https://github.com/szwiep/budE/blob/main/im2ele/estimate.py) for reference) and returns the result.

- Update `boost_urban_aerial.py` according to the _`# YOUR MODEL HERE`_ comments.
- And you're off :rocket: Happy boosting!  

If you have any questions or difficulties setting up `budE` with your model, feel free to reach out to any of the project maintainers!



# Credits

`budE` is a simple pipeline using exciting research done by Miangoleh & Dille et al. in ["_Boosting Monocular Depth Estimation Models to High-Resolution via Content-Adaptive Multi-Resolution Merging_" (2021)](http://yaksoy.github.io/papers/CVPR21-HighResDepth.pdf) and Liu et al. in ["_IM2ELEVATION: Building Height Estimation from Single-View Aerial Imagery_" (2020)](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-12-02719/article_deploy/remotesensing-12-02719.pdf). If `budE` is used in any academic work please cite them:

```
@INPROCEEDINGS{Miangoleh2021Boosting,
author={S. Mahdi H. Miangoleh and Sebastian Dille and Long Mai and Sylvain Paris and Ya\u{g}{\i}z Aksoy},
title={Boosting Monocular Depth Estimation Models to High-Resolution via Content-Adaptive Multi-Resolution Merging},
journal={Proc. CVPR},
year={2021},
}
```
```
IM2ELEVATION: Building Height Estimation from Single-View Aerial Imagery. 
Liu, C.-J.; Krylov, V.A.; Kane, P.; Kavanagh, G.; Dahyot, R. 
Remote Sens. 2020, 12, 2719.
DOI:10.3390/rs12172719
```

The Boosting Monocular Depth Estimation github can be found [here](https://github.com/compphoto/BoostingMonocularDepth), and the IM2ELEVATION github can be found [here](https://github.com/speed8928/IMELE). This pipeline would not be possible if their work was not made available under open-source distribution. Thank you!

