'''
ADM Tronics Image Processing Assignment
Unit test file

Author : Ugurcan Cakal

'''

import os
import cv2
import random
import unittest


import numpy as np
import eye_track as eye
from scipy import signal

data_dir = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = os.path.join(data_dir, 'data')

test_input = os.path.join(os.getcwd(), 'test_input')
test_output = os.path.join(os.getcwd(), 'test_output')

class EyeTrack(unittest.TestCase):

	def test_clip(self):
	    '''
	    Checks if the clipping operation result in expected shape.
	    '''

	    filepath = os.path.join(test_files, eye_rec)
	    cap = cv2.VideoCapture(filepath)

	    while True:
	        ret, frame = cap.read()
	        if not ret:
	            break

	        y = (269,795)
	        x = (537,1416)
	        roi = eye.clip(frame, y, x)
	        self.assertTrue(roi.shape == (np.diff(y), np.diff(x), frame.shape[2]))  
							   
	        # cv2.imshow("FRAME", frame)
	        # cv2.imshow('ROI', roi)

	        # key = cv2.waitKey(30)
	        # if key == 27:
	        #     break

	    cv2.destroyAllWindows()
	    print('Clip test passed\n')

	def test_to_gray(self):
	    '''
	    Checks if gray conversion works well for both grayscale and rgb image.
	    '''

	    filepath = os.path.join(test_files, eye_rec)
	    cap = cv2.VideoCapture(filepath)

	    while True:
	        ret, frame = cap.read()
	        if not ret:
	            break

	        roi = eye.clip(frame, (269,795), (537,1416))
	        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

	        self.assertTrue((eye.to_gray(roi) == eye.to_gray(gray_roi)).all())
		   
	        # cv2.imshow("ROI", eye.to_gray(roi))
	        # cv2.imshow('GRAY ROI', eye.to_gray(gray_roi))

	        # key = cv2.waitKey(30)
	        # if key == 27:
	        #     break

	    cv2.destroyAllWindows()
	    print('Gray test passed\n')

	def test_get_pupiris(self):
		pop = eye.PupilOnFrame()
		for i in range(130,900):
			print(f'Frame {i}')
			filepath = os.path.join(data_dir, 'p1-left', 'frames', f'{i}-eye.png')
			frame = cv2.imread(filepath)
			frame = pop(frame)
			cv2.imshow("frame", frame)
			key = cv2.waitKey(0)

			# break

	def test_get_pupiris2(self):
		in_file = os.path.join(test_input, eye_vid)
		out_file = os.path.join(test_output, 'sub.avi')
		pof = eye.PupirisOnFrame(update_limit=(1,4), radius_range=(48,49), r_step=4, xy_step=8, np_clusters=2, ni_clusters=3, h_factor=1.2, iris=False)
		video_pupil, fps = eye.process_video(in_file, pof, .1)
		eye.save_video(video_pupil, out_file, fps)

	def test_save_input(self):
		pof = eye.PupirisOnFrame(update_limit=(1,4), radius_range=(48,49), r_step=4, xy_step=8, np_clusters=2, ni_clusters=3, h_factor=1.2, iris=False)
		frames = []
		for i in range(0,940):
			print(f'Frame {i}')
			filepath = os.path.join(data_dir, 'p2-right', 'frames', f'{i}-eye.png')
			frame = cv2.imread(filepath)
			frames.append(frame)
		eye.save_video(frames, 'eye_right_2.avi', 24)

	def test_pupiris3(self):
		pof = eye.PupirisOnFrame(update_limit=(1,10), radius_range=(48,49), r_step=4, xy_step=8, np_clusters=2, ni_clusters=3, h_factor=1.2, iris=True)
		for i in range(0,940):
			print(f'Frame {i}')
			filepath = os.path.join(data_dir, 'p2-right', 'frames', f'{i}-eye.png')
			frame = eye.process_image(filepath, pof)
			cv2.imshow("frame", frame)
			key = cv2.waitKey(0)


	def test_hsv(self):
		cap = cv2.VideoCapture(0)
		while True:
			_, frame = cap.read()
			hsv_frame = eye.to_hsv(frame)
			
			# Red color
			low_red = np.array([161, 155, 84])
			high_red = np.array([179, 255, 255])
			red_mask = cv2.inRange(hsv_frame, low_red, high_red)
			red = cv2.bitwise_and(frame, frame, mask=red_mask)

			# Blue color
			low_blue = np.array([94, 80, 2])
			high_blue = np.array([126, 255, 255])
			blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
			blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
			# Green color
			low_green = np.array([25, 52, 72])
			high_green = np.array([102, 255, 255])
			green_mask = cv2.inRange(hsv_frame, low_green, high_green)
			green = cv2.bitwise_and(frame, frame, mask=green_mask)
			# Every color except white
			low = np.array([0, 42, 0])
			high = np.array([179, 255, 255])
			mask = cv2.inRange(hsv_frame, low, high)
			result = cv2.bitwise_and(frame, frame, mask=mask)

			cv2.imshow("Frame", frame)
			cv2.imshow("Red", red)
			cv2.imshow("Blue", blue)
			cv2.imshow("Green", green)
			cv2.imshow("Result", result)
			key = cv2.waitKey(1)
			if key == 27:
				break
		

class RegionEstimate(unittest.TestCase):
	'''
	'''
	def test1_surround_kernel(self):
		'''
		Haar-Like Surround Kernel
		'''
		radius = 1
		surround = eye.HaarSurroundFeature(radius)
		s_kernel = surround.get_kernel()
		print(s_kernel)
		kernel = np.array([[-0.125, -0.125, -0.125, -0.125, -0.125],
						   [-0.125, -0.125, -0.125, -0.125, -0.125],
						   [-0.125, -0.125,  1.,    -0.125, -0.125],
						   [-0.125, -0.125, -0.125, -0.125, -0.125],
						   [-0.125, -0.125, -0.125, -0.125, -0.125]], dtype=np.float32)

		self.assertTrue((kernel == s_kernel).all())

	def test2_region_estima(self):
		'''
		Region estimation for pupil
		detect_pupil function
		'''
		radius = 48
		step = (1,1)
		h_factor = 1.4

		in_file = os.path.join(test_input, eye_img)
		frame = cv2.imread(in_file)

		gray_frame = eye.to_gray(frame)
		row, col = gray_frame.shape
		pad  = 2*radius
		haar_radius = int(h_factor*radius)
		
		# STEP 1 : Get the integral image
		# Need to pad by an additional 1 to get bottom & right edges.
		frame_pad = cv2.copyMakeBorder(gray_frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE) 
		frame_int = cv2.integral(frame_pad)

		# STEP 2 : Convolution
		surround = eye.HaarSurroundFeature(radius)
		filtered, response, center = eye.conv_int(frame_int, surround, step, pad)
		region = eye.get_roi(frame, center, haar_radius)

		# ------------ Test ----------- # 
		fof = frame.copy()
		fof[:,:,2] = filtered

		eye_region = eye.rect_enlarge(region, 2)
		fof = eye.overlay_rectangle(fof, region, center=True)
		fof = eye.overlay_rectangle(fof, eye_region, color=(255,255,0))

		pupil_roi = eye.rect_clip(frame,region)
		eye_roi = eye.rect_clip(frame,eye_region)

		cv2.imshow("frame", frame)
		cv2.imshow("frame_padded", frame_pad)
		cv2.imshow("filtered", filtered)
		cv2.imshow("filter_on_frame", fof)
		cv2.imshow("pupil_roi", pupil_roi)
		cv2.imshow("eye_roi", eye_roi)
		key = cv2.waitKey(0)

class ColorSegment(unittest.TestCase):
	def test_elbow_pupil(self):
		pupil_file = os.path.join(test_input, 'pupil_roi.png')
		pupil_roi = cv2.imread(pupil_file)

		# eye.plt_config(figsize=(16,9),linewidth=1.2,fontsize = 8)
		# eye.kmeans_elbow(pupil_roi, search=(1,14))
		pupil_2 = eye.clustered_frame(pupil_roi, 2)
		pupil_3 = eye.clustered_frame(pupil_roi, 3)
		pupil_4 = eye.clustered_frame(pupil_roi, 4)
		pupil_8 = eye.clustered_frame(pupil_roi, 8)
		cv2.imshow("pupil_roi", pupil_roi)
		cv2.imshow("pupil_2", pupil_2)
		cv2.imshow("pupil_3", pupil_3)
		cv2.imshow("pupil_4", pupil_4)
		cv2.imshow("pupil_8", pupil_8)
		key = cv2.waitKey(0)


	def test_elbow_iris(self):
		iris_file = os.path.join(test_input, 'iris_roi.png')
		iris_roi = cv2.imread(iris_file)

		# eye.plt_config(figsize=(16,9),linewidth=1.2,fontsize = 8)
		# eye.kmeans_elbow(iris_roi, search=(1,14))
		
		iris_2 = eye.clustered_frame(iris_roi, 2)
		iris_3 = eye.clustered_frame(iris_roi, 3)
		iris_4 = eye.clustered_frame(iris_roi, 4)
		iris_8 = eye.clustered_frame(iris_roi, 8)
		cv2.imshow("iris_roi", iris_roi)
		cv2.imshow("iris_2", iris_2)
		cv2.imshow("iris_3", iris_3)
		cv2.imshow("iris_4", iris_4)
		cv2.imshow("iris_8", iris_8)
		key = cv2.waitKey(0)

class PupirisSegment(unittest.TestCase):
	def test1_get_pupil(self):
		'''
		test get pupil
		'''
		pupil_file = os.path.join(test_input, 'pupil_roi.png')
		pupil_roi = cv2.imread(pupil_file)
		i_th= eye.get_threshold(pupil_roi, n_clusters = 2)

		pupiris = None
		kernel = (5,5)
		best_ellipse = ((0,0),(0,0),0)

		opening = cv2.morphologyEx(pupil_roi, cv2.MORPH_OPEN, kernel)
		_, threshold = cv2.threshold(opening, i_th[0], 255, cv2.THRESH_BINARY_INV)
		edges = cv2.Canny(threshold,0,255)

		contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
		
		# ----------- TEST ----------- #

		all_ellipses = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
		

		if len(contours):
			print(len(contours))
			for contour in contours:
				candidate = eye.fit_ellipse_LSQ(contour)
				color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
				all_ellipses = eye.overlay_ellipse(all_ellipses, candidate, color=color)
				if np.sum(candidate[1]) > np.sum(best_ellipse[1]):
					best_ellipse = candidate
					
				pupiris = best_ellipse
		
		pupil_overlay = eye.overlay_ellipse(pupil_roi, best_ellipse, color = (255,255,0))

		ellipse_chosen = np.zeros_like(edges)
		ellipse_chosen = eye.overlay_ellipse(ellipse_chosen, best_ellipse, color = (255,255,0))

		cv2.imshow("roi", pupil_roi)
		cv2.imshow("open", opening)
		cv2.imshow("threshold", threshold)
		cv2.imshow("edges", edges)
		cv2.imshow("all_ellipses", all_ellipses)
		cv2.imshow("best_ellipses", ellipse_chosen)
		cv2.imshow("pupil_overlay", pupil_overlay)

		# cv2.imshow("threshold", threshold)
		key = cv2.waitKey(0)


	def test2_get_iris(self):
		'''
		test get pupiris
		'''
		iris_file = os.path.join(test_input, 'iris_roi.png')
		iris_roi = cv2.imread(iris_file)
		i_th= eye.get_threshold(iris_roi, n_clusters = 3)

		pupiris = None
		kernel = (5,5)
		best_ellipse = ((0,0),(0,0),0)

		opening = cv2.morphologyEx(iris_roi, cv2.MORPH_OPEN, kernel)
		_, threshold = cv2.threshold(opening, i_th[0], 255, cv2.THRESH_BINARY_INV)
		edges = cv2.Canny(threshold,0,255)

		contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
		
		# ----------- TEST ----------- #

		all_ellipses = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

		if len(contours):
			print(len(contours))
			for contour in contours:
				candidate = eye.fit_ellipse_LSQ(contour)
				color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
				all_ellipses = eye.overlay_ellipse(all_ellipses, candidate, color=color)
				if np.sum(candidate[1]) > np.sum(best_ellipse[1]):
					best_ellipse = candidate
					
				pupiris = best_ellipse
		
		iris_overlay = eye.overlay_ellipse(iris_roi, best_ellipse, color = (255,255,0))

		ellipse_chosen = np.zeros_like(edges)
		ellipse_chosen = eye.overlay_ellipse(ellipse_chosen, best_ellipse, color = (255,255,0))

		cv2.imshow("roi", iris_roi)
		cv2.imshow("open", opening)
		cv2.imshow("threshold", threshold)
		cv2.imshow("edges", edges)
		cv2.imshow("all_ellipses", all_ellipses)
		cv2.imshow("best_ellipses", ellipse_chosen)
		cv2.imshow("iris_overlay", iris_overlay)

		# cv2.imshow("threshold", threshold)
		key = cv2.waitKey(0)

	def test3_only_eye(self):
		'''
		test get pupiris
		'''
		radius_range = (32,56)
		i_factor = 2.6

		for i in range(200,940):
			print(f'Frame {i}')
			filepath = os.path.join(data_dir, 'p1-left', 'frames', f'{i}-eye.png')
			frame = cv2.imread(filepath)

			pupil_region,response,radius = eye.radius_search(frame, (48,49), r_step=4, xy_step=8, h_factor=1.4)
			eye_region = eye.rect_enlarge(pupil_region, i_factor, frame.shape)	
			eye_roi = eye.rect_clip(frame,eye_region)

			segmented = eye.clustered_frame(eye_roi, 3)
			# eye_roi = cv2.morphologyEx(eye_roi, cv2.MORPH_OPEN, (9,9))
			# edges = cv2.Canny(eye_roi,0,255)

			cv2.imshow("frame", segmented)
			key = cv2.waitKey(0)


		# cv2.imshow("threshold", threshold)
		key = cv2.waitKey(0)


if __name__=='__main__':
	unittest.main()