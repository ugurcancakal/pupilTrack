'''
ADM Tronics Image Processing Assignment
Unit test file

Author : Ugurcan Cakal

Effort Log
2101091813 2 hours problem exploration
'''

import os
import cv2
import unittest

import re


import numpy as np
import eye_track as eye
from scipy import signal

test_files = os.path.join(os.getcwd(), 'test_files')
data_dir = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = os.path.join(data_dir, 'data')
eye_rec = 'eye_recording.flv'
eye_frame = 'region.png'
eye_frame2 = 'eye.png'

class EyeTrack(unittest.TestCase):

    # def test_clip(self):
    #     '''
    #     Checks if the clipping operation result in expected shape.
    #     '''

    #     filepath = os.path.join(test_files, eye_rec)
    #     cap = cv2.VideoCapture(filepath)

    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break

    #         y = (269,795)
    #         x = (537,1416)
    #         roi = eye.clip(frame, y, x)
    #         self.assertTrue(roi.shape == (np.diff(y), np.diff(x), frame.shape[2]))  
                               
    #         # cv2.imshow("FRAME", frame)
    #         # cv2.imshow('ROI', roi)

    #         # key = cv2.waitKey(30)
    #         # if key == 27:
    #         #     break

    #     cv2.destroyAllWindows()
    #     print('Clip test passed\n')

    # def test_to_gray(self):
    #     '''
    #     Checks if gray conversion works well for both grayscale and rgb image.
    #     '''

    #     filepath = os.path.join(test_files, eye_rec)
    #     cap = cv2.VideoCapture(filepath)

    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break

    #         roi = eye.clip(frame, (269,795), (537,1416))
    #         gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    #         self.assertTrue((eye.to_gray(roi) == eye.to_gray(gray_roi)).all())
           
    #         # cv2.imshow("ROI", eye.to_gray(roi))
    #         # cv2.imshow('GRAY ROI', eye.to_gray(gray_roi))

    #         # key = cv2.waitKey(30)
    #         # if key == 27:
    #         #     break

    #     cv2.destroyAllWindows()
    #     print('Gray test passed\n')

    # def test_initial_region_estimation(self):
    #     '''
    #     '''

    #     filepath = os.path.join(test_files, eye_rec)
    #     cap = cv2.VideoCapture(filepath)

    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break

    #         # y = (269,795)
    #         # x = (537,1416)
    #         # roi = eye.clip(frame, y, x)
    #         # gray_roi = eye.to_gray(roi)
    #         region = eye.initial_region_estimation(frame, 8)

    #         # cv2.imshow("FRAME", frame)
    #         # cv2.imshow('ROI', roi)

    #         # key = cv2.waitKey(30)
    #         # if key == 27:
    #         #     break

    #     cv2.destroyAllWindows()
    #     print('Clip test passed\n')


    # def test_haar_kernel(self):
    #     surround = eye.HaarSurroundFeature(3, val=(0,1))
    #     kernel = surround.get_kernel()
    #     print(kernel)

    #     surround = eye.HaarSurroundFeature(3)
    #     kernel = surround.get_kernel()
    #     print(kernel)

    #     print(np.sum(kernel))

    # def test_conv_int_basic(self):
    #     frame = np.array([[.1,.1,.2,.1],
    #                       [.2,.3,.2,.7],
    #                       [.1,.4,.3,.3],
    #                       [.1,.5,.1,.1]])
    #     r = 1
    #     pad = 2*r
    #     frame_pad = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE) 
    #     frame_int = cv2.integral(frame_pad)

    #     surround = eye.HaarSurroundFeature(r, val= (-1,0))
    #     row, col = frame_pad.shape

    #     frame_conv = np.zeros_like(frame_pad)



    #     for y in range(pad, row-pad, 1):
    #         for x in range(pad, col-pad, 1):
    #             # frame_conv[y-surround.r_out,x-surround.r_out] = eye.get_sum_int(frame_int, surround, pad, y, x)
    #             frame_conv[y,x] = eye.isum_response(frame_int, surround, pad, (y, x))

    #     print("FRAME: \n", frame)
    #     print("SURROND: \n", surround.get_kernel())
    #     print("FRAME PAD: \n", frame_pad)
    #     print("INTEGRAL FRAME: \n", frame_int)
    #     print("FRAME CONVOLVED: \n", frame_conv)


    # def test_int_conv(self):
    #     filepath = os.path.join(test_files, eye_frame)
    #     frame = cv2.imread(filepath)
    #     # frame = eye.to_gray(frame)

    #     padding = 256
        
    #     frame_pad = cv2.copyMakeBorder(eye.to_gray(frame), padding, padding, padding, padding, cv2.BORDER_REPLICATE) 
    #     frame_int = cv2.integral(frame_pad)

    #     surround = eye.HaarSurroundFeature(128)
    #     filtered, _ ,eye_loc = eye.conv_int(frame_int, surround, (2,2), padding)

    #     color = np.copy(frame)
    #     color[:,:,1] = filtered + eye.to_gray(frame)

    #     cv2.imshow("FRAME", eye.get_roi(frame, eye_loc, int(surround.r_in*1.5), 1))
    #     cv2.imshow("FILTERED", eye.get_roi(filtered, eye_loc, surround.r_in))
    #     cv2.imshow("COLOR", eye.get_roi(color, eye_loc, surround.r_in))
    #     key = cv2.waitKey(0)

    # def test_initial_region_estimation(self):

    #     filepath = os.path.join(test_files, eye_frame2)
    #     frame = cv2.imread(filepath)
    #     cv2.imshow("FRAME", eye.initial_region_estimation(frame, radius=32, step=2, h_factor=1, trim=False, outer=True))
    #     key = cv2.waitKey(0)

    # def test_save_video(self):
    #     filepath = os.path.join(data_dir, 'p1-left', 'frames')

    #     filelist = sorted_alphanumeric(os.listdir(filepath))
    #     frame_list = []
    #     for i,filename in enumerate(filelist):
    #         if filename.endswith('.png'):
    #             print(filename)
    #             path = os.path.join(filepath, filename)
    #             frame_list.append(cv2.imread(path))
    #             key = cv2.waitKey(0)

    #     eye.save_video(np.asarray(frame_list))
        
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

if __name__=='__main__':
    unittest.main()