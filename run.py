from operator import getitem
from torchvision.transforms import Compose
from torchvision.transforms import transforms
from torchvision.utils import save_image


# BOOSTING MONOCULAR DEPTH
from boostingMonocularDepth.utils import ImageDataset, calculateprocessingres
from boostingMonocularDepth.boosting import local_boosting, singleestimate, doubleestimate
import boostingMonocularDepth.midas.utils 
from boostingMonocularDepth.midas.models.midas_net import MidasNet
from boostingMonocularDepth.midas.models.transforms import Resize, NormalizeImage, PrepareForNet

# IM2ELE
import im2ele.models.snet as snet # use define_model to get im2ele #TODO: squish define_model into a single class
from im2ele import estimate

# TODO: Should mergenet initialization should be set up/dealt within BoostingMonocularDepth  
# PIX2PIX : MERGE NET
from boostingMonocularDepth.pix2pix.options.test_options import TestOptions
from boostingMonocularDepth.pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel

import time
import os
import torch
import cv2
import numpy as np
import argparse
import warnings
from tqdm import tqdm
warnings.simplefilter('ignore', np.RankWarning)

import logging
logging.getLogger(__name__) # TODO: change prints to logging

# Select device
device = torch.device("cuda")
print("device: %s" % device)

# Global variables
pix2pixmodel = None
im2elemodel = None
factor = None
whole_size_threshold = 700  # TODO!!!: can we figure this out for the project
                            # Maybe find edges - upsize, see how many edges we lose? 
                            # Basically: how far is too far for these images we're using?
GPU_threshold = 1500 - 32 # Limit for the GPU (NVIDIA RTX 2080), can be adjusted 


# MAIN PART OF OUR METHOD
def run(dataset, option):

    if option.Final:
        logging.warning('Performing local boosting via patch estimation. A quick warning: ' \
                                'as-is this method performed poorly for the IM2ELE data which budE '\
                                'based on. It has not yet been optimized for aerial/UAV data.') 

    # Load merge network
    opt = TestOptions().parse()
    global pix2pixmodel
    pix2pixmodel = Pix2Pix4DepthModel(opt)
    pix2pixmodel.save_dir = './boostingMonocularDepth/pix2pix/checkpoints/mergemodel'
    pix2pixmodel.load_networks('latest')
    pix2pixmodel.eval()

    if option.depthNet == 0:
        global im2elemodel
        im2elemodel = snet.define_model()
        state_dict = torch.load('im2ele/Block0_skip_model_110.pth.tar')['state_dict']
        state_dict.pop("E.Harm.dct", None)
        state_dict.pop("E.Harm.weight", None)
        state_dict.pop("E.Harm.bias", None)

        im2elemodel.load_state_dict(state_dict)

        del state_dict
        torch.cuda.empty_cache()
        im2elemodel.to(device)
        im2elemodel.eval()

    # Generating required directories
    result_dir = option.output_dir
    os.makedirs(result_dir, exist_ok=True)

    # TODO: We can get rid of this
    if option.savewholeest:
        whole_est_outputpath = option.output_dir + '_wholeimage'
        os.makedirs(whole_est_outputpath, exist_ok=True)
    if option.savepatchs:
            patchped_est_outputpath = option.output_dir + '_patchest'
            os.makedirs(patchped_est_outputpath, exist_ok=True)
    # TODO: test R20 results
    # Value x of R_x defined in the section 5 of the main paper.
    r_threshold_value = 0.2
    if option.R0:
        r_threshold_value = 0
    elif option.R20:
        r_threshold_value = 0.2

    # TODO: make this quicker -- if we're using one high-res size then we
    # should be able to optimize this. Data loader? Parallel?
    # Go through all images in input directory

    scale_threshold = 2  # Allows up-scaling with a scale up to 2
    # Randomly select X images from the dataset to calculate R_x from Section 6 of 
    # the Boosting Monocular Depth paper:
    procres_sample = np.random.randint(0, len(dataset), size = 1) # TODO: let users parameterize this? If there are disagreements -- then what?
    for sample in procres_sample:
        whole_image_optimal_size, patch_scale = calculateprocessingres(dataset[sample].rgb_image, 
                                                                        option.net_receptive_field_size,
                                                                        r_threshold_value, scale_threshold,
                                                                        whole_size_threshold)

    print(f'\t All wholeImages being processed in: {whole_image_optimal_size}')
    print("start processing")
    # TODO: everything is at the same res so Batch it!!!!!!!
    for image_ind, images in enumerate(tqdm(dataset)):
        # Load image from dataset
        img = images.rgb_image

        path = os.path.join(result_dir, images.name)
        # NOTE: hard-coded because IM2ELE crops the images before estimation
        input_resolution = [440, 440, 3]
        # Typically this would be: input_resolution = img.shape()

        # Generate the base estimate using the double estimation.
      
        whole_estimate = doubleestimate(img, option.net_receptive_field_size, whole_image_optimal_size,
                                        option.pix2pixsize, pix2pixmodel, option.depthNet, im2elemodel, device)
        # TODO: add bilateral filtering to the write_depth process
        if option.R0 or option.R20:
            if option.bilateral_blur:
                whole_estimate = cv2.bilateralFilter(boost, 9, 100, 180) # these params still need to be tinkered with
            boostingMonocularDepth.midas.utils.write_depth(path, cv2.resize(whole_estimate, (input_resolution[1], input_resolution[0]),
                                                    interpolation=cv2.INTER_CUBIC),
                                    bits=2, colored=option.colorize_results)
            continue

        # Output double estimation if required (if not R0/R20 this )
        if option.savewholeest:
            path = os.path.join(whole_est_outputpath, images.name)
            if option.output_resolution == 1:
                midas.utils.write_depth(path,
                                        cv2.resize(whole_estimate, (input_resolution[1], input_resolution[0]),
                                                   interpolation=cv2.INTER_CUBIC), bits=2,
                                        colored=option.colorize_results)
            else:
                midas.utils.write_depth(path, whole_estimate, bits=2, colored=option.colorize_results)

        # The remaining section deals with local boosting via patch estimates on larger resolution 
        # images not constrained by R20 as outlined in Section 6 of Miangoleh & Dille (2021). 
        # 
        # With the IM2ELE UAV images, the local boosting/patch estimation does not work well. 
        # Likely due to the low spatial resolution/quality of UAV images vs. the images BoostingMonocularDepth
        # was built around. However, it is being left in the pipeline as a non-default for 
        #               (1) In the off-chance someone comes along with uncommonly non-noisy aerial data.
        #               (2) As a reminder that this portion can still be investigated/improved in the
        #                        context of boosting UAV/Satellite data. Open source!                    
        if option.Final:
            imageandpatchs, mask_org = local_boosting(option, images, patch_scale, input_resolution, whole_estimate, whole_size_threshold,\
                                                                                        whole_image_optimal_size)
            mask = mask_org.copy()
            # Enumerate through all patches, generate their estimations and refining the base estimate.
            for patch_ind in range(len(imageandpatchs)):
                
                #### TODO: put this in a func get_patch_information
                # Get patch information
                patch = imageandpatchs[patch_ind] # patch object
                patch_rgb = patch['patch_rgb'] # rgb patch
                patch_whole_estimate_base = patch['patch_whole_estimate_base'] # corresponding patch from base
                rect = patch['rect'] # patch size and location
                patch_id = patch['id'] # patch ID
                org_size = patch_whole_estimate_base.shape # the original size from the unscaled input
                print('\t processing patch', patch_ind, '|', rect)
                ######


                # We apply double estimation for patches. The high resolution value is fixed to twice the receptive
                # field size of the network for patches to accelerate the process.
                patch_estimation = doubleestimate(patch_rgb, option.net_receptive_field_size, option.patch_netsize,
                                                option.pix2pixsize, pix2pixmodel, option.depthNet, im2elemodel, device)

                # Output patch estimation if required
                if option.savepatchs:
                    path = os.path.join(patchped_est_outputpath, imageandpatchs.name + '_{:04}'.format(patch_id))
                    boostingMonocularDepth.midas.utils.write_depth(path, patch_estimation, bits=2, colored=option.colorize_results)

                ##### TODO: put this into function merge_patch_estimations()
                patch_estimation = cv2.resize(patch_estimation, (option.pix2pixsize, option.pix2pixsize),
                                            interpolation=cv2.INTER_CUBIC)

                patch_whole_estimate_base = cv2.resize(patch_whole_estimate_base, (option.pix2pixsize, option.pix2pixsize),
                                                    interpolation=cv2.INTER_CUBIC)

                # Merging the patch estimation into the base estimate using our merge network:
                # We feed the patch estimation and the same region from the updated base estimate to the merge network
                # to generate the target estimate for the corresponding region.
                pix2pixmodel.set_input(patch_whole_estimate_base, patch_estimation)

                # Run merging network
                pix2pixmodel.test()
                visuals = pix2pixmodel.get_current_visuals()

                prediction_mapped = visuals['fake_B']
                prediction_mapped = (prediction_mapped+1)/2
                prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

                mapped = prediction_mapped

                # We use a simple linear polynomial to make sure the result of the merge network would match the values of
                # base estimate
                p_coef = np.polyfit(mapped.reshape(-1), patch_whole_estimate_base.reshape(-1), deg=1)
                merged = np.polyval(p_coef, mapped.reshape(-1)).reshape(mapped.shape)

                merged = cv2.resize(merged, (org_size[1],org_size[0]), interpolation=cv2.INTER_CUBIC)

                # Get patch size and location
                w1 = rect[0]
                h1 = rect[1]
                w2 = w1 + rect[2]
                h2 = h1 + rect[3]

                # To speed up the implementation, we only generate the Gaussian mask once with a sufficiently large size
                # and resize it to our needed size while merging the patches.
                if mask.shape != org_size:
                    mask = cv2.resize(mask_org, (org_size[1],org_size[0]), interpolation=cv2.INTER_LINEAR)

                tobemergedto = imageandpatchs.estimation_updated_image

                # Update the whole estimation:
                # We use a simple Gaussian mask to blend the merged patch region with the base estimate to ensure seamless
                # blending at the boundaries of the patch region.
                tobemergedto[h1:h2, w1:w2] = np.multiply(tobemergedto[h1:h2, w1:w2], 1 - mask) + np.multiply(merged, mask)
                imageandpatchs.set_updated_estimate(tobemergedto)
                ##### Should return imageandpatchs.estimation_updated_image 

                # Output the result
                path = os.path.join(option.output_dir, imageandpatchs.name)

                boostingMonocularDepth.midas.utils.write_depth(path,
                                    cv2.resize(imageandpatchs.estimation_updated_image,
                                    (input_resolution[1], input_resolution[0]),
                                    interpolation=cv2.INTER_CUBIC), bits=2, colored=option.colorize_results)
                                           
            
    print("finished")


    

if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, required=True, help='input files directory '
                                                                    'Images can be .png .jpg .tiff')
    parser.add_argument('--output_dir', type=str, required=True, help='result dir. result depth will be png.'
                                                                      ' vides are JMPG as avi')
    parser.add_argument('--savepatchs', type=int, default=1, required=False,
                        help='Activate to save the patch estimations')
    parser.add_argument('--savewholeest', type=int, default=0, required=False,
                        help='Activate to save the base estimations')
    parser.add_argument('--net_receptive_field_size', type=int, required=False)  # Do not set the value here
    parser.add_argument('--pix2pixsize', type=int, default=1024, required=False)  # Do not change it
    parser.add_argument('--depthNet', type=int, default=0, required=False,
                        help='use to select different base depth networks 0:midas 1:strurturedRL 2:LeRes')
    parser.add_argument('--colorize_results', action='store_true')
    parser.add_argument('--R0', action='store_true')
    parser.add_argument('--R20', action='store_true')
    parser.add_argument('--Final', action='store_true')
    parser.add_argument('--max_res', type=float, default=np.inf)
    parser.add_argument('--bilateral_blur', type=int, default=0, required=False)

    # Check for required input
    option_, _ = parser.parse_known_args()
    # print(option_) # 
    if int(option_.R0) + int(option_.R20) + int(option_.Final) == 0:
        assert False, 'Please activate one of the [R0, R20, Local] options using --[R0]'
    elif int(option_.R0) + int(option_.R20) + int(option_.Final) > 1:
        assert False, 'Please activate only ONE of the [R0, R20, Final] options'

    # Setting each networks receptive field and setting the patch estimation resolution to twice the receptive
    # field size to speed up the local refinement as described in the section 6 of the main paper.
    if option_.depthNet == 0:
        option_.net_receptive_field_size = 440 # Training resolution of IM2ELE
        option_.patch_netsize = 2 * option_.net_receptive_field_size
    else:
        assert False, 'depthNet can currently only be 0'

    # Create dataset from input images
    dataset_ = ImageDataset(option_.data_dir, 'test')

    # Run pipeline
    run(dataset_, option_)
