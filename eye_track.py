'''
ADM Tronics Image Processing Assignment

Author : Ugurcan Cakal

'''

import sys
import cv2
import time
import progressbar

import numpy as np
import random as rng
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from cv2 import VideoWriter, VideoWriter_fourcc


# UTILITY CLASSES AND FUNCTIONS  

# ----------- Prerequisites ----------- # 

class HaarSurroundFeature:
	'''
	Haar like feature detection kernel 
	Store the inner and outer values in the sparse form.
	Return the dense form if required.
	'''

	def __init__(self, r_inner, r_outer=None, val=None):
		'''
		 _________________
		|        -ve      |
		|     _______     |
		|    |   +ve |    |
		|    |   .   |    |
		|    |_______|    |
		|         <r1>    |
		|_________<--r2-->|

		Frobenius normalized values
		
		Want norm = 1 where norm = sqrt(sum(pixelvals^2)), so:
		sqrt(count_inner*val_inner^2 + count_outer*val_outer^2) = 1
		
		Also want sum(pixelvals) = 0, so:
		count_inner*val_inner + count_outer*val_outer = 0
		
		Solving both of these gives:
		val_inner = sqrt(count_outer/(count_inner*count_outer + sq(count_inner)) );
		val_outer = sqrt(count_inner/(count_inner*count_outer + sq(count_outer)) );

		Square radius normalised values
		
		Want the response to be scale-invariant, so scale it by the number of pixels inside it:
		val_inner = 1/count = 1/r_outer^2
		
		Also want sum(pixelvals) = 0, so:
		count_inner*val_inner + count_outer*val_outer = 0

		Arguments:

			r_inner(int):
				inner radius of the kernel, r1.

			r_outer(int):
				outer radius of the kernel, r2.
				Default:None 
				In the default case, r2 = 3*r

			val(tuple of float):
				inner and outer values to be assigned
				Default:None
				In the default case, values are calculated using
				Frobenius Normalization process.
		'''

		if r_outer is None:
			r_outer = r_inner*3

		count_inner = r_inner*r_inner;
		count_outer = r_outer*r_outer - r_inner*r_inner

		if val is None:
			val_inner = 1.0 / (r_inner*r_inner)
			val_outer = -val_inner*count_inner/count_outer

		else:
			val_inner = val[0];
			val_outer = val[1];

		self.val_in = val_inner
		self.val_out= val_outer
		self.r_in = r_inner
		self.r_out = r_outer
	
	def get_kernel(self):
		'''
		Get the kernel in the dense matrix form
		For debugging
		'''
		kernel = np.zeros(shape=(2*self.r_out, 2*self.r_out))
		kernel.fill(self.val_out)

		start = (self.r_out-self.r_in)
		end = start + 2*self.r_in

		kernel[start:end, start:end] = self.val_in
		return kernel

# ------------------------------------- #



# ----------- Preprocessing ----------- # 

def get_roi(frame, center, radius):
	'''
	Rectangular region of interest given the frame 
	depending on the given radius and the center

	Arguments:
		frame(np.ndarray of dim = 2 or 3):
			the frame to processed

		center(tuple of int):
			x,y coordinate of the center of region of interest

		radius(int):
			radius of the region of interest, 

	Returns:
		rectangle(tuple of tuple of int):
			opencv type rectangle definition
			(x_start, y_start), (x_end, y_end)
	'''
	start = tuple([val-radius for val in center])
	end = tuple([val+radius for val in center])
	return (start, end)

def initial_region_estimation(frame, radius, step, h_factor=1.5):
	'''
	!!! ASSUMPTION : Pupil and iris together can be roughly described as
	"a dark blob surrounded by a light background"

	To find the pupil, use Haar-like feature detector on the integral image.
	Integral image reduces the time complexity from O(N^2) to O(N)

	The Haar-like feature is square, and so is only an approximation 
	to the elliptical pupil shape.

		Arguments:

			frame(np.ndarray of dim = 2 or 3):
				the frame to processed

			radius(int):
				inner radius for the HaarSurroundFeature kernel

			step(tuple of int):
				ystep, xstep
				stepsize for kernel walk

			h_factor(float):
				factor of increase in calculating 
				haar_radius = h_factor*radius
				Default=1.5

		Returns:

			region(tuple of tuple of int):
				inital region estimation in the form
				of a opencv rectangle 
				((x_start, y_start), (x_end, y_end))
	'''

	# Make step a tuple in case a single number is passed
	if not isinstance(step, (tuple, list)):
		if isinstance(step, int):
			step = (step, step)
		else:
			info = 'Unsupported step respresentation\n'
			info += 'Example : step = (2,2)'
			raise ValueError(info)

	# frame = to_gray(frame)
	row, col = to_gray(frame).shape
	pad  = 2*radius
	haar_radius = int(h_factor*radius)
	
	# STEP 1 : Get the integral image
	# Need to pad by an additional 1 to get bottom & right edges.
	frame_pad = cv2.copyMakeBorder(to_gray(frame), pad, pad, pad, pad, cv2.BORDER_REPLICATE) 
	frame_int = cv2.integral(frame_pad)

	# STEP 2 : Convolution
	surround = HaarSurroundFeature(radius)
	filtered, _, center = conv_int(frame_int, surround, step, pad)
	region = get_roi(frame, center, haar_radius)

	return region	


## ----------- Integral Convolution ----------- ##

def conv_int(frame_int, kernel, step, padding):
	'''
	Convolution on an integral image.

		-----------------------
		Find best haar response
		-----------------------

					_____________________
				   |         Haar kernel |
				   |                     |
		 __________|______________       |
		| Image    |      |       |      |
		|    ______|______|___.-r-|--2r--|
		|   |      |      |___|___|      |
		|   |      |          |   |      |
		|   |      |          |   |      |
		|   |      |__________|___|______|
		|   |    Search       |   |
		|   |    region       |   |
		|   |                 |   |
		|   |_________________|   |
		|                         |
		|_________________________|

		Arguments:
			frame_int(np.ndarray of dim=2):
				integral frame

			kernel(HaarSurroundFeature object):
				surround kernel to be used in the convolution

			step(tuple of int):
				ystep, xstep
				stepsize for kernel walk

			padding(int):
				number of padded pixels 

		Returns:

			frame_conv(np.ndarray of dim=2):
				a frame consisting of kernel responses
			
			response(int):
				the strongest kernel response

			center(tuple of int): 
				normalized pixel position of the strongest response

	'''
	# Init
	row, col = frame_int.shape
	row -= 1
	col -= 1
	y_min = 0 
	x_min = 0
	min_response = 255
	y_step, x_step = step
	f_shape = row - 2*padding, col- 2*padding
	
	frame_conv = np.zeros(shape = f_shape, dtype=np.uint8)    

	# Convolution on the integral frame
	for y in range(padding, row-padding, y_step):
		for x in range(padding, col-padding, x_step):
			response = isum_response(frame_int, kernel, padding, (y, x))
			# Keep track of minimum response
			if response<min_response:
				min_response = response
				y_min = y - padding
				x_min = x - padding
			frame_conv[y-padding,x-padding] = response

	center = (x_min,y_min)

	return frame_conv, response, center

def isum_response(frame_int, kernel, padding, pos):
	'''
	Kernel response in the respected position calculated on the integral image
	p00+p11-p01-p10 gives the area sum on the ingtegral image. 
	p11 -> inside the desired area
	p00, p10, p01 -> neighbours

		¦         ¦ 
		|         |  p00._____________________.p01
		|         |     |         Haar kernel |
		|         |     |                     |
		|         |     |   p00._______.p01   |
		|-padding-|     |      |       |      |
		|         |     |      | (x,y) |      |
		|         |     |      |_______|      |
		|         |     |   p10'       'p11   |
		|         |     |                     |
		|         |     |_____________________|
		|         |  p10'                     'p11
		¦         ¦

		Arguments:

			frame_int(np.ndarray of dim=2):
				integral frame

			kernel(HaarSurroundFeature object):
				surround kernel to be used in the convolution

			padding(int):
				number of padded pixels

			pos(tuple of int):
				represent y,x position

		Returns:
			response(int):
				Kernel response of the respective pixel

	'''
	# Init
	ylim, xlim = frame_int.shape
	ylim -= 1
	xlim -= 1
	y, x = pos
	r_in = kernel.r_in
	r_out = kernel.r_out
	
	# Inner area sum
	inner_sum = frame_int[y-r_in,x-r_in] + frame_int[y+r_in, x+r_in] - frame_int[y-r_in, x+r_in] - frame_int[y+r_in, x-r_in]
	
	# Outer area tends not to fit the padded image. 
	# Make sure that no indexing error will be raised
	p00 = y-r_out if y-r_out>0 else 0, x-r_out if x-r_out>0 else 0
	p11 = y+r_out if y+r_out<ylim else ylim, x+r_out if x+r_out<xlim else xlim
	p01 = y-r_out if y-r_out>0 else 0, x+r_out if x+r_out<xlim else xlim
	p10 = y+r_out if y+r_out<ylim else ylim, x-r_out if x-r_out>0 else 0
	outer_sum = frame_int[p00] + frame_int[p11] - frame_int[p01] - frame_int[p10] - inner_sum

	# Calculate the response (O(N) :))
	response = kernel.val_in*inner_sum + kernel.val_out*outer_sum
	
	return response

## -------------------------------------------- ##
# ------------------------------------- # 



# ----------- Pupil Segmentation ----------- # 

def get_threshold(frame):
	'''
	The dark cluster is assumed to correspond to the pupil pixels,
	a segmented binary image of the pupil region is created by 
	thresholding any pixels above the maximum intensity in the dark cluster.

	Instead of this manual parameter setting, the wish is to have a 
	fully automatic threshold calculation, which adapts to illumination changes.

	The approach is to segment the image histogram into two
	clusters, corresponding to pupil and background intensity values.
	We use k-means clustering on the histogram of the pupil region
	to find two clusters: dark and light. The dark cluster
	is then assumed to correspond to the pupil pixels, and a segment of
	binary image of the pupil region is created by thresholding any
	pixels above the maximum intensity in the dark cluster.

		Arguments:
			frame(np.ndarray of dim = 2 or 3):
				the frame of interest

		Returns:
			threshold(int):
				adaptive threshold pixel value
	'''
	flat_frame = frame.reshape(-1, 1)
	kmeans = KMeans(n_clusters=2, random_state=0)
	kmeans.fit(flat_frame)
	labels = kmeans.labels_

	# Maximum intensity in the dark cluster
	threshold = flat_frame[labels==0].max()

	return threshold

def get_contours(frame_roi, th, th_max=255, gaussian_kernel=(7,7)):
	'''
	Extract the candidate contours
	,which are to be used in ellipse fitting,
	from binary thresholded image

		Arguments:

			frame_roi(np.ndarray of dim = 2):
				grayscale region of interest which harbor the pupil
			
			th(int):
				pixel threshold value
			
			th_max(int):
				maximum pixel value to use with the THRESH_BINARY
			
			gaussion_kernel(tuple of int):
				kernel size of the gauissian blur filter

		Returns:
			contours(list of dim = 4):
				list of contours found using canny edge detection on
				binary thresholded image.
				sorted by area.
	'''

	_, threshold = cv2.threshold(frame_roi, th, th_max, cv2.THRESH_BINARY)
	threshold = cv2.GaussianBlur(threshold, gaussian_kernel, 0)
	edges = cv2.Canny(threshold,0,255)

	contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
	contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
	return contours

def fit_ellipse_LSQ(candidate):
	'''
	Simple least-squares ellipse fit to boundary points

		Arguments:
			candidate(np.ndarray(n,2) of dtype int):
				Candidate pupil-iris boundary points from edge detection

		Returns:
			best_ellipse(tuple of tuples of float)
				Best fitted ellipse parameters 
				((x_center, y_center), (major_ax,minor_ax), theta)
	
	'''
	best_ellipse = ((0,0),(1e-6,1e-6),0)

	# Break if too few points to fit ellipse (RARE)
	if candidate.shape[0] < 5:
		return best_ellipse

	best_ellipse = cv2.fitEllipse(candidate)

	return best_ellipse

def pupil_segmentation(frame_roi):
	'''
	Segment the pupil using an adaptive threshold (coming from K-Means)
	Extract the candidate contours from the binary thresholded image
	Fit an ellipse to the best candidate	

		Arguments:
			frame_roi(np.ndarray of dim = 2):
				grayscale region of interest which harbor the pupil
		
		Returns:
			pupil_ellipse(tuple of tuples of float):
				the parameters of the ellipse overlaying the pupil
				((x_center, y_center), (major_ax,minor_ax), theta)
	
	'''
	pupil_ellipse = None
	th = get_threshold(frame_roi)
	contours = get_contours(frame_roi, th)
	if len(contours):
		pupil_ellipse = fit_ellipse_LSQ(contours[0])

	return pupil_ellipse

# ------------------------------------------ # 



# ----------- Frame Manipulation ----------- # 

def to_gray(frame):
	'''
	Reduce the channels to 1(grayscale)

		Arguments:

			frame(np.ndarray of dim = 2 or 3):
				the frame to be converted to grayscale

		Return:

			frame(np.ndarray of dim = 2):
				grayscale frame
	'''
	if len(frame.shape)==2:
		return frame
	if (len(frame.shape)==3) and frame.shape[2] == 3:
		return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	elif (len(frame.shape)==3) and frame.shape[2] == 4:
		return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
	else:
		raise ValueError('Unsupported number of channels')

def clip(frame, y, x):
	'''
	Clip the full frame to given x and y coordinates 
	to get a smaller region of interest

		Arguments:

			frame(np.ndarray of dim = 2 or 3):
				the frame to be clipped

			y(tuple of int):
				(ymin, ymax): y coordinates of 
				region of interestin the y axis

			x(tuple of int):
				(xmin, xmax) x coordinates of 
				region of interestin the y axis
		
		Returns:
			frame(np.ndarray of dim = 2 or 3):
				clipped frame

	'''
	return frame[y[0]:y[1], x[0]:x[1]]

def clip_rect(frame, rectangle):
	'''
	Clip the full frame to given rectangular frame
	Preserves the original frame
		Arguments:

			frame(np.ndarray of dim = 2 or 3):
				the frame to be clipped

			rectangle(tuple of tuple of int):
				opencv type rectangle definition
				(x_start, y_start), (x_end, y_end)

		Returns:
			frame(np.ndarray of dim = 2 or 3):
				clipped frame

	'''
	start, end = rectangle
	y = (start[1], end[1])
	x = (start[0], end[0])

	return clip(frame, y, x)


## ----------- Overlay ----------- ##

def overlay_rectangle(frame, rectangle, offset=None, color=(0, 255, 0), thickness=2, preserve=True):
	'''
	Helper funtion to overlay a rectangle on a frame

		Arguments:
			frame(np.ndarray of dim = 2 or 3):
				the frame to be processed

			rectangle(tuple of tuple of int):
				opencv type rectangle definition
				(x_start, y_start), (x_end, y_end)

			color(tuple of int)
				BGR color of the rectangle highlight
				Used when trim option is false
				Default=(0, 255, 0) (green)

			thickness(int):
				Line thickness of the rectangle highlight
				Used when trim option is false
				Default=2

			preserve(bool):
				preserve the origial frame or not
				Default=True

		Returns:
			frame(np.ndarray of dim = 2 or 3):
				frame with rectangle overlay

	'''
	if preserve:
		return cv2.rectangle(frame.copy(), *rectangle, color, thickness)
	else:
		return cv2.rectangle(frame, *rectangle, color, thickness)

def overlay_pupil(frame, ellipse, offset=None, color=(0, 255, 0), thickness=2, center=True, preserve=True):
	'''
	Overlay the pupil on the frame given the pupil ellipse

		Arguments:
			frame(np.ndarray of dim = 2 or 3):
				the frame to be processed

			ellipse(tuple of tuples of float):
				the parameters of the ellipse overlaying the pupil
				((x_center, y_center), (major_ax,minor_ax), theta)

			offset(tuple of int):
				the offset amount calculated from the top left corner
				(x,y)
				Default = None

			color(tuple of int)
				BGR color of the ellipse highlight
				Used when trim option is false
				Default=(0, 255, 0) (green)

			thickness(int):
				Line thickness of the ellipse highlight
				Used when trim option is false
				Default=2

			center(bool):
				highlight the center or not
				Default=True

			preserve(bool):
				preserve the origial frame or not
				Default=True

		Returns:
			frame(np.ndarray of dim = 2 or 3):
				frame with pupil overlay

	'''
	if offset is not None:
		ellipse = ellipse_offset(ellipse,offset)

	if preserve:
		f2 = cv2.ellipse(frame.copy(), ellipse, color, thickness)
	else:
		f2 = cv2.ellipse(frame, ellipse, color, thickness)

	center_coordinates = np.array(np.round(ellipse[0]),dtype=np.int)
	if center:
		f2 = cv2.circle(f2, tuple(center_coordinates), 1, color, thickness)

	return f2

def ellipse_offset(ellipse, offset):
	'''
	Offset the ellipse from the center in the x,y plane given offset amount

		Arguments:	
			ellipse(tuple of tuples of float):
				the parameters of the ellipse
				((x_center, y_center), (major_ax,minor_ax), theta)

			offset(tuple of int):
				the offset amount calculated from the top left corner
				(x,y)
		
		Returns:
			ellipse(tuple of tuples of float):
				the parameters of the resulting ellipse
				((x_center, y_center), (major_ax,minor_ax), theta)
			
	'''
	center_coordinates = np.array(ellipse[0]) + np.array(offset)
	ellipse = (tuple(center_coordinates), ellipse[1], ellipse[2])
	return ellipse

## ------------------------------- ##
# ------------------------------------------ # 



# ----------- Utility ----------- # 

def save_video(frame_seq,filepath='test.avi',fps=24):
	'''
	Save the given frame sequence in the form of a video

		Arguments:

			frame_seq(list of np.ndarrays of dim = 3):
				list of frames to be saved as a video

			filepath(string):
				destination path
				Default = 'test.avi'

			fps(int):
				frame per second

	'''
	tic = time.perf_counter()
	fourcc = VideoWriter_fourcc(*'MP42')
	frame_shape = frame_seq[0].shape
	video = VideoWriter(filepath, fourcc, float(fps), (frame_shape[1], frame_shape[0]))  

	print(f"{filepath} saving...")
	widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
	bar = progressbar.ProgressBar(maxval=len(frame_seq), widgets = widgets).start()

	for i, frame in enumerate(frame_seq):
		video.write(frame)
		bar.update(i)
	video.release()
	bar.finish()

	toc = time.perf_counter()
	print(f"{filepath} saved in {toc-tic:0.4f} seconds!\n")

# ------------------------------- # 


# ----------- Main ----------- # 

def process_image(in_file):
	frame = cv2.imread(in_file)
	frame = pupil_on_frame(frame)
	return frame

def pupil_on_frame(frame):
	'''
	Process single image and overlay pupil
	'''
	# STEP 1 : Convert the frames to the grayscale version
	gray_roi = to_gray(frame)

	# STEP 2 : Approximate the pupil region to reduce the search space
	region = initial_region_estimation(gray_roi,radius=32, step=4, h_factor=1.5)
	roi = clip_rect(gray_roi, region)   

	# STEP 4: Pupil Segmentation
	pupil = pupil_segmentation(roi)
	if pupil is not None:
		frame = overlay_pupil(frame, pupil, offset=region[0])
	return frame

def process_video(in_file, percentage=1.0):
	'''
	Process a video and overlay pupil
	'''
	# Init
	i = 0
	tic = time.perf_counter()
	video = cv2.VideoCapture(in_file)
	fps = video.get(cv2.CAP_PROP_FPS)
	frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
	video_pupil = [None]*int(frame_count*percentage)
	
	# Iterate through the frames in the video
	print(f"{in_file} processing...")
	widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
	bar = progressbar.ProgressBar(maxval=int(frame_count), widgets = widgets).start()

	for i,frame in enumerate(video_pupil):
		ret, frame = video.read()
		if not ret:
			break
		video_pupil[i] = pupil_on_frame(frame)
		bar.update(i)

	bar.finish()
	toc = time.perf_counter()
	print(f"{in_file} processed in {toc-tic:0.4f} seconds!\n")

	return video_pupil, fps

def main(mode, in_file, out_file, percentage=1.0):
	'''
	'''
	if mode == 'image':
		image_pupil = process_image(in_file)
		cv2.imwrite(out_file, image_pupil)

	elif mode == 'video':
		video_pupil, fps = process_video(in_file, percentage)
		save_video(video_pupil, out_file, fps)

	else:
		info = 'Unsupported mode of operation!\n'
		info += 'Refer to user manual by eye_track -h'
		raise ValueError(info)

	'''
	# Main operation
	# '''
	# cap = cv2.VideoCapture(filepath)

	# while True:
	# 	ret, frame = cap.read()
	# 	if not ret:
	# 		break

	# 	# STEP 1 : Clip the frames such that they only include 1 eye 
	# 	# roi = clip(frame, (269,795), (537,1416))

	# 	# STEP 2 : Convert the frames to the grayscale version
	# 	gray_roi = to_gray(frame)

	# 	# STEP 3 : Approximate the pupil region to reduce the search space
	# 	region = initial_region_estimation(gray_roi,radius=32, step=4, h_factor=1.5)
	# 	roi = clip_rect(gray_roi, region)   

	# 	# STEP 4: Pupil Segmentation
	# 	pupil = pupil_segmentation(roi)
	# 	if pupil is not None:
	# 		segment = overlay_pupil(frame, pupil, offset=region[0])

	# 	cv2.imshow("REGION", segment)
	# 	# cv2.imshow("ROI", roi)
	# 	key = cv2.waitKey(30)
	# 	if key == 27:
	# 		break

	# cv2.destroyAllWindows()


	# cap = cv2.VideoCapture('rec2.avi')

	# while(cap.isOpened()):
	# 	ret, frame = cap.read()

	# 	# cv2.imshow('frame',frame)

		# # STEP 2 : Convert the frames to the grayscale version
		# gray_roi = to_gray(frame)

		# # STEP 3 : Approximate the pupil region to reduce the search space
		# region = initial_region_estimation(gray_roi,radius=36, step=8, h_factor=1.5)
		# roi = clip_rect(gray_roi, region)   

		# # STEP 4: Pupil Segmentation
		# pupil = pupil_segmentation(roi)
		# if pupil is not None:
		# 	segment = overlay_pupil(frame, pupil, offset=region[0])

	# 		cv2.imshow("REGION", segment)

	# 	if cv2.waitKey(1) & 0xFF == ord('q'):
	# 		break

	# # cap.release()
	# cv2.destroyAllWindows()
# ---------------------------- # 

def parse_argv(argv):
	'''
	Parse the argument vector
	Example 
	python3 eye_track.py -m video -i eye_recording.flv -o eye_rec_pupil.avi

		Arguments:
			argv(list of str):
				command line arguments passed to the main function

		Returns:
			mode(str):
				operating mode of the system

			in_file(str):
				input filepath

			out_file(str):
				output filepath

	'''	
	mode, in_file, out_file, percentage = None, None, None, 1.0

	for i,arg in enumerate(argv[1:],1):
		# Output
		if arg ==  '-h':
			print(user_manual())
		# Mode
		if arg ==  '-m':
			mode = argv[i+1]
		# Input
		if arg ==  '-i':
			in_file = argv[i+1]
		# Output
		if arg ==  '-o':
			out_file = argv[i+1]
		# Percentage
		if arg ==  '-p':
			percentage = float(argv[i+1])

	return mode, in_file, out_file, percentage

def user_manual():
	'''
	User manual as a string
		Returns:
			info(str):
				user manual
	'''
	info = "\n\n # ------- Eye Tracker User Manual ------- # \n\n"
	info += "Command Line Arguments:\n\n"
	info += "-m <operation mode>\n"
	info += "i.e. image/video\n\n"
	info += "-i <input file>\n"
	info += "i.e. eye_rec.avi\n\n"
	info += "-o <output file>\n"
	info += "i.e. eye_rec_pupil.avi\n\n"

	return info

if __name__ == '__main__':

	if len(sys.argv)>1:
		mode, in_file, out_file, percentage = parse_argv(sys.argv)
		if mode is not None:
			main(mode, in_file, out_file, percentage)

	# print(sys.argv)