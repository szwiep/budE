''' Scratch-pad file used in the development of the budE pipeline. 

Most logic concerned with exploring the SSIM of boost vs. 
non-boosted depth estimations on the IM2ELE network. This file 
is kept in the budE project source-code to maintain a record
of at least some of the methods used to justify the simplifications
in the project structure.
'''
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import imageio
from skimage.measure import compare_ssim
import cv2
import os

import boostingMonocularDepth.midas.utils

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from skimage.metrics import structural_similarity

def show_structural_differences(gray_single, gray_boost):
    # # SSIM thresholding to find very different items
    # # https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python

    # Compute SSIM between two images
    (score, diff) = structural_similarity(gray_single, gray_boost, full=True)
    print("Image similarity", score)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1] 
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.drawContours(mask, [c], 0, (0,255,0), -1)
            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

    cv2.imshow('before', before)
    cv2.imshow('after', after)
    cv2.imshow('diff',diff)
    cv2.imshow('mask',mask)
    cv2.imshow('filled after',filled_after)
    cv2.waitKey(0)


def compare_SSIM(single, boost, ground_truth):
    # Simple SSIM comparison 
    # https://ourcodeworld.com/articles/read/991/how-to-calculate-the-structural-similarity-index-ssim-between-two-images-with-python

    # 5. Compute the Structural Similarity Index (SSIM) between the two
    #    images, ensuring that the difference image is returned
    (score, diff) = structural_similarity(single, ground_truth, full=True)
    diff = (diff * 255).astype("uint8")

    (boost_score, boost_diff) = structural_similarity(boost, ground_truth, full=True)
    diff = (boost_diff * 255).astype("uint8")

    return score, boost_score

def join_depth_estimations(images):
    ''' A naive join of NxN depth estimation images into 
        larger depth estimation with x'*N by y'*N.
        

        TODO: Find ground elevation in im1. Join bottom and right 
        layers to im1. Shift top and bottom layers elev according to 
        elev in im1. Gaussian blur on border to smooth transition.

        Args:
            images: a list of image matrices to be joined. They will
                    be joined in linear order. For example an input
                    of [im1, im2, im3, im4] will become a joined image:     
                                
                                    im1 im3
                                    im2 im4
        Returns:
            joined_depth: (2*N)x(2*N) depth estimation
    ''' 
    # Take the smallest values in the top-left image to be the ground value:
    # TODO: this does not work 
    ground_value = np.min(images[0])

    # mask = np.zeros(images[0].shape)
    # ground_indices = (images[0] < 50)
    # print(np.count_nonzero(ground_indices))
    # mask[ground_indices] = 1
    # print(images[0], ground_value)

    # cv2.imshow('ground_mask', mask)
    # cv2.waitKey(0)
    reference_image = images[0]
    reference_image = reference_image.reshape((-1,1))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label,center=cv2.kmeans(reference_image, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    cv.imshow('res2',res2)
    cv.waitKey(0)
    cv.destroyAllWindows()
        
    # IM2ELE RGB are 500x500, depth estimations are 440x440 centre crops
    # RGB overlaps 250 so depth estimations will overlap 130


    # TODO: first things first, just join them side by side... find numbering pattern

def center_crop(img, dim): 
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

if __name__ == '__main__':

    single_ssims = np.zeros(367)
    boost_ssims = np.zeros(367)
    blurred_boost_ssims = np.zeros(367)
    filenames = []

    # TODO: Analyze the SSIM for basic blurs. How does the boosts hold up?
    #           is the SSIM for boosts or blurred boosts better? What are 
    
    patch_count = 0
    image_list = []
    # for iter, filename in enumerate(os.listdir('test_rgbs_base')):  
    #     patch_count += 1 
    img_path_1 = f'/media/szwiep/LaCie/im2ele-idvres/boost_max_700_R0/boosted/315000_233500_RGB_0_9.png'
    img_path_2 = f'/media/szwiep/LaCie/im2ele-idvres/boost_max_700_R0/boosted/315000_233500_RGB_1_9.png'
    img_1 = cv2.imread(img_path_1)
    img_2 = cv2.imread(img_path_2)  
    image_list.append(img_1)
    image_list.append(img_2)

    join_depth_estimations(image_list)


    # for iter, filename in enumerate(os.listdir('test_heights_adjust')):

    #     RGB_filename = filename.replace('height', 'RGB')
    #     ground_truth_path = f'test_heights_adjust/{filename}'
    #     boost_path = f'outputs/max_700_R0_all/{RGB_filename}'
    #     low_est_filename = RGB_filename.replace('.png', '')
    #     single_path = f'outputs/max_700_R0_all/{low_est_filename}low_est.png'


    #     single = cv2.imread(single_path)
    #     boost = cv2.imread(boost_path)
    #     ground_t = cv2.imread(ground_truth_path)

    #     # Convert images to grayscale
    #     single_gray = cv2.cvtColor(single, cv2.COLOR_BGR2GRAY)
    #     boost_gray = cv2.cvtColor(boost, cv2.COLOR_BGR2GRAY)
    #     ground_truth = cv2.cvtColor(ground_t, cv2.COLOR_BGR2GRAY)

    #     # Crop the ground truth to the required size... IM2ELE stuff
    #     ground_truth = center_crop(ground_truth, (440,440))

    #     ##########################################################
    #     # Bilateral filtering stuff
    #     #######################################################
        
    #     blurred_boost = cv2.bilateralFilter(boost, 9, 100, 180)
    #     # cv2.imshow('boost', boost)
    #     # cv2.imshow('blurred boost', blurred_boost)
    #     # cv2.waitKey(0)

    #     blurred_gray = cv2.cvtColor(blurred_boost, cv2.COLOR_BGR2GRAY)
    #     midas.utils.write_depth(f'outputs/max_700_R0_all/{low_est_filename}_blur', blurred_boost, bits=2, colored=False)

    #     ##########################################################
    #     # SSIM Stuff (How has the structure changes)
    #     ##########################################################
    
    #     single_ssim, boost_ssim = compare_SSIM(single_gray, boost_gray, ground_truth)
    #     _, blur_boost_ssim = compare_SSIM(single_gray, blurred_gray, ground_truth)
    
    #     filenames.append(filename)
    #     single_ssims[iter] = single_ssim
    #     boost_ssims[iter] = boost_ssim
    #     blurred_boost_ssims[iter] = blur_boost_ssim
    

    # d = {'file': filenames,  'no_boost_ssim': single_ssims, 
    #                             'boost_ssim': boost_ssims, 
    #                             'blur_boost_ssim': blurred_boost_ssims}
    # boost_ssim_frame = pd.DataFrame(data=d)
    # boost_ssim_frame.to_csv('./boost_ssim_frame_100_180.csv')

    # show_structural_differences(single_gray, boost_gray)
