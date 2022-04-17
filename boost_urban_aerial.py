from operator import getitem
from torchvision.transforms import Compose
from torchvision.transforms import transforms
from torchvision.utils import save_image


# BOOSTING MONOCULAR DEPTH
from boostingMonocularDepth.utils import ImageDataset, rgb2gray
from boostingMonocularDepth.boosting import local_boosting, get_patch_information, singleestimate, \
                                                    doubleestimate, merge_patch_estimations, \
                                                    calculateprocessingres, load_merge_network
import boostingMonocularDepth.midas.utils 
from boostingMonocularDepth.midas.models.midas_net import MidasNet
from boostingMonocularDepth.midas.models.transforms import Resize, NormalizeImage, PrepareForNet

# IM2ELE
import im2ele.models.snet as snet # use define_model to get im2ele #TODO: squish define_model into a single class
from im2ele import estimate
from im2ele.estimate import estimateim2ele


import time
import os
import torch
import cv2
import numpy as np
import argparse
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
warnings.simplefilter('ignore', np.RankWarning)

import logging
logging.getLogger(__name__)

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print("Using device: %s" % device)

# Global variables
pix2pixmodel = None
im2elemodel = None
GPU_threshold = 1500 - 32 # Limit for the GPU (NVIDIA RTX 2080), can be adjusted 

def boost_urban_aerial(dataset, option):
    ''' The budE pipeline. Composed of 3 main steps:
            (1) (a) Load the Aerial/Satellite DE models (ADE)
                (b) Load Boosting Monocular Depth merge model (BMD)
            (2) Estimate the aerial/satellite images with ADE at low and 
                    high resolutions then merge with BMD
            (3) Storing boosted images for analysis/viewing pleasure!
        
        The budE pipeline is based on the pipeline created in the 
        Boosting Monocular Depth project (run.run) but with some simplifications
        to help speed up the process and work better with aerial data. 

        Want to add your own aerial/satellite DE model in? Awesome! The sections of 
        the code that you'll need to edit are marked with '#YOUR MODEL CODE HERE'
        and are followed by an example of what should be added.

        This pipeline is not complete! There are still improvements to be made
        and more work to be investigated. Specifically in v.1.0.0 we're still 
        curious whether boostingMonocularDepth.boosting.calculateprocessingres 
        can be optimized for aerial data (even more so if satellite, at nadir).
    '''

    if option.local_boost:
        logging.warning('Performing local boosting via patch estimation. A quick warning: ' \
                                'this (as-is) method performed poorly for the IM2ELE data which budE '\
                                'is based on. It has not yet been optimized for aerial/UAV data.') 
    if option.depthNet == 0:
        global im2elemodel
        im2elemodel = snet.define_model()
        print('Loading pre-trained IM2ELE model...')
        state_dict = torch.load('im2ele/Block0_skip_model_110.pth.tar')['state_dict']
        state_dict.pop("E.Harm.dct", None)
        state_dict.pop("E.Harm.weight", None)
        state_dict.pop("E.Harm.bias", None)

        im2elemodel.load_state_dict(state_dict)

        del state_dict
        torch.cuda.empty_cache()
        im2elemodel.to(device)
        im2elemodel.eval()
    # YOUR MODEL CODE HERE (load your model w/ weights and set for eval)
    # if option.depthNet == 1:
    #   your_model_path = `path_to_where_your_pt_is.pt`
    #   global yourmodel
    #   yourmodel = YourModel(your_model_path.pt)
    #   youmodel.to(device)
    #   yourmodel.eval()

    global whole_size_threshold
    if option.interactive_max:
        img = dataset[0]
        img_rgb = img.rgb_image
        whole_size_threshold = interactive_max_threshold(img_rgb)
        torch.cuda.empty_cache()
    else:
        whole_size_threshold = 700 # Hard-coded choice. Taken from experimentation with IM2ELE
    print(f'Maximum high resolution set as: {whole_size_threshold}\n')

    global pix2pixmodel
    pix2pixmodel = load_merge_network('./boostingMonocularDepth/pix2pix/checkpoints/mergemodel', 'latest')

    # Generating required directories
    result_dir = option.output_dir
    os.makedirs(result_dir, exist_ok=True)
    
    # Value x of R_x defined in the section 5 of the main paper.
    r_threshold_value = 0.2
    if option.R0:
        r_threshold_value = 0
    elif option.R20:
        r_threshold_value = 0.2

    # This first section will take the input images, estimate their depth with the provided aerial/satellite
    # depth estimation model (default IM2ELE) first at low-res then at the high-res, and then will merge
    # the two low-res and high-res estimations using BoostingMonocularDepth's merging network.
    #
    # This pipeline's structure is largely borrow from the Boosting Monocular Depth's pipeline but with 
    # some simplications allowed by the use-case of noisy/small UAV and satellite data. The most notable 
    # chages are that the high-res value is only computed once for the whole dataset, and that users 
    # have an option to apply a bilateral filter to the boosted depth estimation to mitigate noise 
    # introduced by up-sampling in  the boosting process.

    scale_threshold = 2.5  # Allow up-scaling up to double the image size (IM2ELE goes _sharply_ down hill after this)

    # Randomly select X images from the dataset to calculate R_x from Section 6 of the Boosting Monocular 
    # Depth paper: http://yaksoy.github.io/papers/CVPR21-HighResDepth.pdf
    procres_sample = np.random.randint(0, len(dataset), size = 1) 
    for sample in procres_sample:
        whole_image_optimal_size, patch_scale = calculateprocessingres(dataset[sample].rgb_image, 
                                                                        option.net_receptive_field_size,
                                                                        r_threshold_value, scale_threshold,
                                                                        whole_size_threshold)
    # NOTE: hard-coded because IM2ELE crops the images as part of the data transform
    input_resolution = [440, 440, 3] # Typically this would be: input_resolution = img.shape()
    print(f'Double estimation process using {input_resolution[0]} as low-res and {whole_image_optimal_size} as high-res.')
    print(f'Beginning depth estimation for boosting...')
    for image_ind, images in enumerate(tqdm(dataset)):
        # Load image from dataset
        img = images.rgb_image
        path = os.path.join(result_dir, images.name)
        
        # Generate the base estimate using the double estimation.
        whole_estimate = doubleestimate(img, option.net_receptive_field_size, whole_image_optimal_size,
                                        option.pix2pixsize, pix2pixmodel, option.depthNet, im2elemodel, device)
     
        # Save the boosted depth estimation and exit pipeline.
        if not option.local_boost:
            if option.bilateral_blur:
                whole_estimate = cv2.bilateralFilter(whole_estimate, 9, 100, 180) # Can still be tinkered with. Based on analyze.py results
            boostingMonocularDepth.midas.utils.write_depth(path, cv2.resize(whole_estimate, (input_resolution[1], input_resolution[0]),
                                                    interpolation=cv2.INTER_CUBIC),
                                    bits=2, colored=option.colorize_results)    
            continue
        # The remaining section deals with local boosting via patch estimates on larger resolution 
        # images not constrained by R20 as outlined in Section 6 of Miangoleh & Dille (2021). 
        #
        # With the IM2ELE UAV images, the local boosting/patch estimation does not work well. This is
        # likely due to both the larger signal/noise ratio present in UAV/Satellite images as well 
        # as the training of the IM2ELE network. However, local boosting is being left in the pipeline as 
        # a non-default parameter for the following reasons:
        #               (1) In the chance someone (maybe you!) comes along with a better network or 
        #                       aerial data that is uncommonly not-very-noisy
        #               (2) As a reminder that this portion can still be investigated/improved in the
        #                        context of boosting UAV/Satellite data. Open source!                    
        if option.local_boost:
            imageandpatchs, mask_org = local_boosting(option, images, patch_scale, input_resolution, whole_estimate, whole_size_threshold,\
                                                                                        whole_image_optimal_size)
            # Enumerate through all patches, generate their estimations, and refine the base estimate by merging estimation into base estimate.
            for patch_ind in range(len(imageandpatchs)):
                patch_rgb, patch_whole_estimate_base, rect, patch_id, org_size = get_patch_information(patch_ind, imageandpatchs)

                estimation_updated_img = merge_patch_estimations(option, patch_estimation, patch_whole_estimate_base, imageandpatchs, \
                                                                            pix2pixmodel, mask_org, rect, org_size)

                # Output the result
                path = os.path.join(option.output_dir, imageandpatchs.name)
                boostingMonocularDepth.midas.utils.write_depth(path,
                                    cv2.resize(estimation_updated_img,
                                    (input_resolution[1], input_resolution[0]),
                                    interpolation=cv2.INTER_CUBIC), bits=2, colored=option.colorize_results)
                                           
    print(f'Boosting of aerial/satellite urban depth estimation complete. See results in {result_dir}')

def interactive_max_threshold(rgb_image):
    ''' Produces a plot of depth estimations at different resolutions using 
    the base model then asks user to select a resolution with CLI input.

    The purpose of this function to to allow a user to make a visually informed
    (but not combersome) choice about the whole_size_threshold parameter. The 
    quality of the IM2ELE images/depth estimations is not comparable to those
    used in the Boosting Monocular Depth project so a max_size_threshold of
    3000 does not work for boosting. 
    '''
    # TODO: alter this function so that users select the max_threshold by 
    #           clicking on the plot. 
    resolutions = np.arange(rgb_image.shape[0], rgb_image.shape[0]+501, 100)
    
    print('\nInteractively setting whole_size_threshold for the BoostingMonocularDepth pipeline..')
    fig, axs = plt.subplots(1, 6)
    for i, res in enumerate(resolutions):
        depth_est = estimateim2ele(rgb_image, res, im2elemodel, device)
        axs[i].imshow(depth_est, cmap='gray')
        if res == rgb_image.shape[0]:
            axs[i].set_title(f'input res') 
        else: axs[i].set_title(f'res={res}')   

    for a in axs:
        a.set_xticklabels([])
        a.set_yticklabels([])

    fig.suptitle('Depth estimations from IM2ELE with different input resolutions (res)')
    plt.subplots_adjust(wspace=0, hspace=5)
    plt.show(block=False)

    print( 'whole_size_threshold will set the limit for how large the input image can be upsampled to '\
            ' in the boosting process (high-res will never exceed it). See the figure for an example '\
            ' of how your input images behave with the depth estimation model at different input resolutions.\n ')
    choice = None
    while choice is None:
        try:
            selection = int(input(f'Select one of {resolutions[1:]} as the max threshold: '))
        except ValueError:
            print(f'"{selection}" is not in {resolutions}.')
        if selection not in resolutions:    
            print(f'{selection} is not in {resolutions}.')
        else:
            confirmation = input(f'Select {selection} as the max threshold? (Y)es or (N)o? ')
            if confirmation == 'Y':
                choice = int(selection)
                plt.close()
    return choice


if __name__ == "__main__":
    # This is mostly borrowed from Boosting Monocular Depth's run __name__ entry-point!
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, required=True, help='input files directory '
                                                                    'Images can be .png .jpg .tiff')
    parser.add_argument('--output_dir', type=str, required=True, help='result dir. result depth will be png.'
                                                                      ' vides are JMPG as avi')
    parser.add_argument('--net_receptive_field_size', type=int, required=False)  # Do not set the value here
    parser.add_argument('--pix2pixsize', type=int, default=1024, required=False)  # Do not change it
    parser.add_argument('--depthNet', type=int, default=0, required=False,
                        help='Use to select different base depth networks 0:IM2ELE 1:Not implemented yet. Could be yours!')
    parser.add_argument('--colorize_results', action='store_true')
    parser.add_argument('--R0', action='store_true')
    parser.add_argument('--R20', action='store_true')
    parser.add_argument('--local_boost', action='store_true')
    parser.add_argument('--bilateral_blur',  action='store_true')
    parser.add_argument('--interactive_max',  action='store_true')

    # Check for required input
    option_ = parser.parse_args()
    if int(option_.R0) + int(option_.R20) + int(option_.local_boost) == 0:
        option_.R20 = 1 # Default is R20
    elif int(option_.R0) + int(option_.R20) + int(option_.local_boost) > 1:
        assert False, 'Please activate only ONE of the [R0, R20, Final] options'

    # Setting each networks receptive field and setting the patch estimation resolution to twice the receptive
    # field size to speed up the local refinement as described in the section 6 of Boosting Monocular Depth's main paper.
    if option_.depthNet == 0:
        option_.net_receptive_field_size = 440 # Training resolution of IM2ELE
        option_.patch_netsize = 2 * option_.net_receptive_field_size
    # YOUR MODEL CODE HERE (set as CLI arg)
    # if option.depthNet == 1:
    #   option_.net_receptive_field_size = your_models_receptive_field 
    #   option_.patch_netsize = 2 * option_.net_receptive_field_size
    else:
        assert False, 'depthNet can currently only be 0'

    # Create dataset from input images
    dataset_ = ImageDataset(option_.data_dir, 'test')

    # Run pipeline
    boost_urban_aerial(dataset_, option_)
