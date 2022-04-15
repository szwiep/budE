from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import imageio
from skimage.measure import compare_ssim
import cv2
import os

import midas.utils

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from skimage.metrics import structural_similarity


# def get_original_boost_difference(single, boost, rgb):
#     '''
#         single: NumPy array of first (orignial) depth estimation
#         boost: NumPy array of original depth estimation
#         rgb: NumPy array of original RGB image
#     '''

#     diff = np.abs(boost - single)
#     print(diff)

    
#     fig, axs = plt.subplots(1, 3)
#     fig.suptitle('Horizontally stacked subplots')
#     axs[0].imshow(single)
#     axs[1].imshow(boost)
#     axs[2].imshow(diff, cmap='jet')
    
#     # fig = Figure()
#     # canvas = FigureCanvas(fig)
#     # ax1 = fig.add_subplot(1, 3, 1)
#     # ax1.imshow(single)

#     # ax2 = fig.add_subplot(1, 3, 2)
#     # ax2.imshow(boost)

#     # ax3 = fig.add_subplot(1, 3, 3)
#     # ax3.imshow(diff, cmap='gist_heat')
#     # fig.show()
#     plt.show()

def show_structural_differences(gray_single, gray_boost):
    
    # single = 'outputs/max_700_R0_all/315000_233500_RGB_5_2.png'
    # boost = 'outputs/max_700_R0_all/315000_233500_RGB_5_2low_est.png'

    # # SSIM thresholding to find very different items
    # # https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python
    # before = cv2.imread(single)
    # after = cv2.imread(boost)

    # # Convert images to grayscale
    # before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    # after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

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


    # 6. You can print only the score if you want
    # print("SSIM single: {}, SSIM boost: {}".format(score, boost_score))

    return score, boost_score


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

    # TODO: loop over all the images to get SSIM + Blur results for all!
    # TODO: Analyze the SSIM for basic blurs. How does the boosts hold up?
    #           is the SSIM for boosts or blurred boosts better? What are 
    #           the largest/smallest SSIM pairs?
    for iter, filename in enumerate(os.listdir('test_heights_adjust')):

        RGB_filename = filename.replace('height', 'RGB')
        ground_truth_path = f'test_heights_adjust/{filename}'
        boost_path = f'outputs/max_700_R0_all/{RGB_filename}'
        low_est_filename = RGB_filename.replace('.png', '')
        single_path = f'outputs/max_700_R0_all/{low_est_filename}low_est.png'


        single = cv2.imread(single_path)
        boost = cv2.imread(boost_path)
        ground_t = cv2.imread(ground_truth_path)

        # Convert images to grayscale
        single_gray = cv2.cvtColor(single, cv2.COLOR_BGR2GRAY)
        boost_gray = cv2.cvtColor(boost, cv2.COLOR_BGR2GRAY)
        ground_truth = cv2.cvtColor(ground_t, cv2.COLOR_BGR2GRAY)

        # Crop the ground truth to the required size... IM2ELE stuff
        ground_truth = center_crop(ground_truth, (440,440))

        ##########################################################
        # Bilateral filtering stuff
        #######################################################
        
        blurred_boost = cv2.bilateralFilter(boost, 9, 100, 180)
        # cv2.imshow('boost', boost)
        # cv2.imshow('blurred boost', blurred_boost)
        # cv2.waitKey(0)

        blurred_gray = cv2.cvtColor(blurred_boost, cv2.COLOR_BGR2GRAY)
        midas.utils.write_depth(f'outputs/max_700_R0_all/{low_est_filename}_blur', blurred_boost, bits=2, colored=False)

        ##########################################################
        # SSIM Stuff (How has the structure changes)
        ##########################################################
    
        single_ssim, boost_ssim = compare_SSIM(single_gray, boost_gray, ground_truth)
        _, blur_boost_ssim = compare_SSIM(single_gray, blurred_gray, ground_truth)
    
        filenames.append(filename)
        single_ssims[iter] = single_ssim
        boost_ssims[iter] = boost_ssim
        blurred_boost_ssims[iter] = blur_boost_ssim
    

    d = {'file': filenames,  'no_boost_ssim': single_ssims, 
                                'boost_ssim': boost_ssims, 
                                'blur_boost_ssim': blurred_boost_ssims}
    boost_ssim_frame = pd.DataFrame(data=d)
    boost_ssim_frame.to_csv('./boost_ssim_frame_100_180.csv')

    # show_structural_differences(single_gray, boost_gray)
