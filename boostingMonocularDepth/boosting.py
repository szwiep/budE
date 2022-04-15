from operator import getitem
from torchvision.transforms import Compose
from torchvision.transforms import transforms
from torchvision.utils import save_image

from im2ele.estimate import estimateim2ele

import boostingMonocularDepth.midas.utils 
from boostingMonocularDepth.utils import generatemask, ImageandPatchs, rgb2gray, applyGridpatch, getGF_fromintegral

import numpy as np
import torch
import cv2
import os

# TODO: remove os. calls from here. That should be handled by main pipeline!
def local_boosting(option, images, patch_scale, input_resolution, whole_estimate, whole_size_threshold, whole_image_optimal_size):
    # Compute the multiplier described in section 6 of the main paper to make sure our initial patch can select
    # small high-density regions of the image.
    img = images.rgb_image
    mask_org = generatemask((whole_size_threshold, whole_size_threshold)) # TODO: This probably doesn't hold for UAV/Satellite quality

    global factor
    factor = max(min(1, 4 * patch_scale * whole_image_optimal_size / whole_size_threshold), 0.2)
    print('Adjust factor is:', 1/factor)

    # Check if Local boosting is beneficial.
    if option.max_res < whole_image_optimal_size:
        print("No Local boosting. Specified Max Res is smaller than R20")  
        path = os.path.join(result_dir, images.name)
        if option.output_resolution == 1:
            midas.utils.write_depth(path,
                                    cv2.resize(whole_estimate,
                                                (input_resolution[1], input_resolution[0]),
                                                interpolation=cv2.INTER_CUBIC), bits=2,
                                    colored=option.colorize_results)
        else:
            midas.utils.write_depth(path, whole_estimate, bits=2,
                                    colored=option.colorize_results)
        return

    # Compute the default target resolution.
    if img.shape[0] > img.shape[1]:
        a = 2 * whole_image_optimal_size
        b = round(2 * whole_image_optimal_size * img.shape[1] / img.shape[0])
    else:
        a = round(2 * whole_image_optimal_size * img.shape[0] / img.shape[1])
        b = 2 * whole_image_optimal_size
    b = int(round(b / factor))
    a = int(round(a / factor))

    # recompute a, b and saturate to max res.
    if max(a,b) > option.max_res:
        print('Default Res is higher than max-res: Reducing final resolution')
        if img.shape[0] > img.shape[1]:
            a = option.max_res
            b = round(option.max_res * img.shape[1] / img.shape[0])
        else:
            a = round(option.max_res * img.shape[0] / img.shape[1])
            b = option.max_res
        b = int(b)
        a = int(a)

    img = cv2.resize(img, (b, a), interpolation=cv2.INTER_CUBIC)

    # Extract selected patches for local refinement
    base_size = option.net_receptive_field_size*2
    patchset = generatepatchs(img, base_size)

    print('Target resolution: ', img.shape)

    # Computing a scale in case user prompted to generate the results as the same resolution of the input.
    # Notice that our method output resolution is independent of the input resolution and this parameter will only
    # enable a scaling operation during the local patch merge implementation to generate results with the same resolution
    # as the input.
    # if option.output_resolution == 1:
    # NOTE: there is not output_resolution arg in budE. buDE will always output to the input res 
    mergein_scale = input_resolution[0] / img.shape[0]
    #     print('Dynamicly change merged-in resolution; scale:', mergein_scale)
    # else:
    #     mergein_scale = 1

    imageandpatchs = ImageandPatchs(option.data_dir, images.name, patchset, img, mergein_scale)
    whole_estimate_resized = cv2.resize(whole_estimate, (round(img.shape[1]*mergein_scale),
                                        round(img.shape[0]*mergein_scale)), interpolation=cv2.INTER_CUBIC)
    imageandpatchs.set_base_estimate(whole_estimate_resized.copy())
    imageandpatchs.set_updated_estimate(whole_estimate_resized.copy())

    print('\t Resulted depthmap res will be :', whole_estimate_resized.shape[:2])
    print('patchs to process: '+str(len(imageandpatchs)))

    return imageandpatchs, mask_org


# Adaptively select patches
def adaptiveselection(integral_grad, patch_bound_list, gf):
    patchlist = {}
    count = 0
    height, width = integral_grad.shape

    search_step = int(32/factor)

    # Go through all patches
    for c in range(len(patch_bound_list)):
        # Get patch
        bbox = patch_bound_list[str(c)]['rect']

        # Compute the amount of gradients present in the patch from the integral image.
        cgf = getGF_fromintegral(integral_grad, bbox)/(bbox[2]*bbox[3])

        # Check if patching is beneficial by comparing the gradient density of the patch to
        # the gradient density of the whole image
        if cgf >= gf:
            bbox_test = bbox.copy()
            patchlist[str(count)] = {}

            # Enlarge each patch until the gradient density of the patch is equal
            # to the whole image gradient density
            while True:
                bbox_test[0] = bbox_test[0] - int(search_step/2)
                bbox_test[1] = bbox_test[1] - int(search_step/2)

                bbox_test[2] = bbox_test[2] + search_step
                bbox_test[3] = bbox_test[3] + search_step

                # Check if we are still within the image
                if bbox_test[0] < 0 or bbox_test[1] < 0 or bbox_test[1] + bbox_test[3] >= height \
                        or bbox_test[0] + bbox_test[2] >= width:
                    break

                # Compare gradient density
                cgf = getGF_fromintegral(integral_grad, bbox_test)/(bbox_test[2]*bbox_test[3])
                if cgf < gf:
                    break
                bbox = bbox_test.copy()

            # Add patch to selected patches
            patchlist[str(count)]['rect'] = bbox
            patchlist[str(count)]['size'] = bbox[2]
            count = count + 1
    
    # Return selected patches
    return patchlist

# Generating local patches to perform the local refinement described in section 6 of the main paper.
def generatepatchs(img, base_size):
    
    # Compute the gradients as a proxy of the contextual cues.
    img_gray = rgb2gray(img)
    whole_grad = np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)) +\
        np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3))

    threshold = whole_grad[whole_grad > 0].mean()
    whole_grad[whole_grad < threshold] = 0

    # We use the integral image to speed-up the evaluation of the amount of gradients for each patch.
    gf = whole_grad.sum()/len(whole_grad.reshape(-1))
    grad_integral_image = cv2.integral(whole_grad)

    # Variables are selected such that the initial patch size would be the receptive field size
    # and the stride is set to 1/3 of the receptive field size.
    blsize = int(round(base_size/2))
    stride = int(round(blsize*0.75))

    # Get initial Grid
    patch_bound_list = applyGridpatch(blsize, stride, img, [0, 0, 0, 0])

    # Refine initial Grid of patches by discarding the flat (in terms of gradients of the rgb image) ones. Refine
    # each patch size to ensure that there will be enough depth cues for the network to generate a consistent depth map.
    print("Selecting patchs ...")
    patch_bound_list = adaptiveselection(grad_integral_image, patch_bound_list, gf)

    # Sort the patch list to make sure the merging operation will be done with the correct order: starting from biggest
    # patch
    patchset = sorted(patch_bound_list.items(), key=lambda x: getitem(x[1], 'size'), reverse=True)
    return patchset


# Generate a double-input depth estimation
def doubleestimate(img, size1, size2, pix2pixsize, pix2pixmodel, net_type, depthmodel, device):
    # Generate the low resolution estimation
    estimate1 = singleestimate(img, size1, net_type, depthmodel, device)
    # Resize to the inference size of merge network.
    estimate1 = cv2.resize(estimate1, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

    # Generate the high resolution estimation
    estimate2 = singleestimate(img, size2, net_type, depthmodel, device)
    # Resize to the inference size of merge network.
    estimate2 = cv2.resize(estimate2, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

    # Inference on the merge model
    with torch.no_grad():
        pix2pixmodel.set_input(estimate1, estimate2)
        pix2pixmodel.test()
        visuals = pix2pixmodel.get_current_visuals()
        prediction_mapped = visuals['fake_B']
        prediction_mapped = (prediction_mapped+1)/2
        prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / (
                    torch.max(prediction_mapped) - torch.min(prediction_mapped))
        prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

    return prediction_mapped


# Generate a single-input depth estimation
def singleestimate(img, msize, net_type, model, device):
    # if msize > GPU_threshold:
    #     print(" \t \t DEBUG| GPU THRESHOLD REACHED", msize, '--->', GPU_threshold)
    #     msize = GPU_threshold

    if net_type == 0:
        return estimateim2ele(img, msize, model, device)
    # Add new model here:
    # e.g if your_model == 1:
    #       return estimateYourModel(img, msize)
