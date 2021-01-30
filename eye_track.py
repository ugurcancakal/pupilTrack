'''
ADM Tronics Image Processing Assignment

Author : Ugurcan Cakal

'''
import re
import sys
import cv2
import time
import progressbar

import numpy as np
import random as rng
import matplotlib.pyplot as plt

from cv2 import VideoWriter, VideoWriter_fourcc
from matplotlib import rc
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# ----------- Section 1 : Pre-Processing ----------- # 

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

def to_hsv(frame):
	'''
	Move the frame to HSV color space

		Arguments:
			frame(np.ndarray of dim = 3):
				the frame to be converted to HSV color space

		Return:
			frame(np.ndarray of dim = 3):
				HSV frame
	'''
	if len(frame.shape)==2:
		frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
		return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	if (len(frame.shape)==3) and frame.shape[2] == 3:
		return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	elif (len(frame.shape)==3) and frame.shape[2] == 4:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
		return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

## ----------- Section 1.1 : Rectangle Operations ----------- ##

def rect_clip(frame, rectangle):
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

def rect_list_to_tuple(rect_xywh):
	'''
	Convert a rectangle defined in the form of 
	[x_start, y_start, width, height] to 
	((x_start,y_start),(x_end,y_end))
	
		Arguments:
			rect_xywh(list of int):
				[x_start, y_start, width, height]

		Returns:
			rect_x0y0x1y1(tuple of tuple of int):
				opencv type rectangle definition
				(x_start, y_start), (x_end, y_end)

	'''
	start = tuple(rect_xywh[0:2])
	end = tuple(rect_xywh[0:2] + rect_xywh[2:])
	return (start,end)

def rect_enlarge(rectangle, factor, frame_shape=None):
	'''
	Enlarge a rectangle keeping the center the same

		Arguments:
			rectangle(tuple of tuple of int):
				opencv type rectangle definition
				(x_start, y_start), (x_end, y_end)

			factor(float):
				enlarge factor
				if factor = 2, the width and height of the 
				rectangle will be doubled but 
				the center will be the same.

			frame_shape(tuple of int):
				the shape of the frame in which the rectangle
				to be fitted. used for dimension check.
				in the case the enlarged rectangle goes beyond the 
				frame dimensions, it will be clipped.
				Default = None

		Returns:
			rectangle(tuple of tuple of int):
				enlarged rectangle in opencv type rectangle definition
				(x_start, y_start), (x_end, y_end)
	'''
	start = rectangle[0]
	end = rectangle[1]

	x_en = (end[0] - start[0])*(factor-1)//2
	y_en = (end[1] - start[1])*(factor-1)//2

	start = [int(start[0]-x_en), int(start[1]-y_en)]
	end = [int(end[0]+x_en), int(end[1]+y_en)]

	# Check if the rectangle fits the frame
	if frame_shape:
		# Start Check
		if start[0] < 0:
			start[0] = 0
		if start[1] < 0:
			start[1] = 0

		# End Check
		if end[0] >= frame_shape[1]:
			end[0] = frame_shape[1]

		if end[1] >= frame_shape[0]:
			end[1] = frame_shape[0]
	
	return (tuple(start),tuple(end))

# ----------- Section 2 : Post-Processing ----------- #

## ----------- Section 2.1 : Status ----------- ##

def overlay_status(frame, status_dict, color = (0, 255, 0), thickness = 1, font = cv2.FONT_HERSHEY_SIMPLEX, fontScale = .5):
	'''
	Append the status information provided to the bottom of the frame. 

		Arguments:
			frame(np.ndarray of dim = 3):
				the frame to be appended with status

			status_dict(dict):
				a dictionary including of any information to be presented. 
			
			color(tuple of int)
				BGR color of text string to be drawn. 
				Default=(0, 255, 0) (green)

			thickness(int): 
				The thickness of the line in px

			font(int): It denotes the font type. 
				All the available fonts in OpenCV are as follows
				FONT_HERSHEY_SIMPLEX = 0
				FONT_HERSHEY_PLAIN = 1
				FONT_HERSHEY_DUPLEX = 2
				FONT_HERSHEY_COMPLEX = 3
				FONT_HERSHEY_TRIPLEX = 4
				FONT_HERSHEY_COMPLEX_SMALL = 5
				FONT_HERSHEY_SCRIPT_SIMPLEX = 6
				FONT_HERSHEY_SCRIPT_COMPLEX = 7

			fontScale(flaot): 
				Font scale factor that is multiplied by 
				the font-specific base size.

		Returns:
			vis(np.ndarray of dim = 3):
				the frame concatenated with status
	'''
	shape = ((len(status_dict)+1)*20, frame.shape[1] ,frame.shape[2])
	status = np.zeros(shape=shape, dtype = np.uint8)

	org = [20,20]

	if status_dict:
		for i, (key,value) in enumerate(status_dict.items()):
			stat_line = f'{key} : '
			if isinstance(value, (tuple, list)):
				stat_line+=tuple_to_string(value)
			else:
				stat_line+= f'{value:.1f}'
			
			status = cv2.putText(status, stat_line, tuple(org), font,  
						   fontScale, color, thickness, cv2.LINE_AA)
			org[1] += 20 

		vis = np.concatenate((frame, status), axis=0)

	else:
		return frame

	return vis

def tuple_to_string(tup, precison=1):
	'''
	Represent a tuple as a string with defined precision

		Arguments:
			tup(tuple):
				the tuple to be presented

			precision(int):
				Number of digits of precision for floating point output

		Returns:
			stat_line(str):
				string representation of the tuple	
	'''
	stat_line= '('
	for i,val in enumerate(tup):
		if isinstance(val, (tuple, list)):
			stat_line+=tuple_to_string(val)
		else:
			stat_line+= f'{val:.{precison}f}'

		if i<len(tup)-1:
			stat_line+= ' , '
		else:
			stat_line+= ')'
	return stat_line

## ----------- Section 2.2 : Rectangle ----------- ##

def overlay_rectangle(frame, rectangle, center=False, color=(0, 255, 0), thickness=2, preserve=True):
	'''
	Helper funtion to overlay a rectangle on a frame

		Arguments:
			frame(np.ndarray of dim = 2 or 3):
				the frame to be processed

			rectangle(tuple of tuple of int):
				opencv type rectangle definition
				(x_start, y_start), (x_end, y_end)

			center(bool):
				highlight the center or not
				Default=False

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
		return  cv2.rectangle(frame, *rectangle, color, thickness)

	# if center:
	# 	start,end = rectangle
	# 	center_coordinates = ((end[0] + start[0])//2, (end[1] + start[1])//2)
	# 	rect = cv2.circle(rect, tuple(center_coordinates), 1, color, thickness)

	# return rect


## ----------- Section 2.3 : Ellipse ----------- ##

def overlay_ellipse(frame, ellipse, offset=None, color=(0, 255, 0), thickness=2, center=True, preserve=True):
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

	
	if center:
		center_coordinates = np.array(np.round(ellipse[0]),dtype=np.int)
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

# ----------- Section 3 : Utility ----------- # 

def save_video(frame_seq,filepath='test.avi',fps=24):
	'''
	Save the given frame sequence in the form of a video
	Use the shape of the biggest frame in the frame_seq list

		Arguments:
			frame_seq(list of np.ndarrays of dim = 3):
				list of frames to be saved as a video

			filepath(string):
				destination path
				Default = 'test.avi'

			fps(int):
				frame per second
	'''
	shape_idx = np.argmax([frame.shape[0] for frame in frame_seq])
	frame_shape = frame_seq[shape_idx].shape

	tic = time.perf_counter()
	fourcc = VideoWriter_fourcc('M','J','P','G')
	video = VideoWriter(filepath, fourcc, float(fps), (frame_shape[1], frame_shape[0]))  

	print(f"{filepath} saving...")
	widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
	bar = progressbar.ProgressBar(maxval=len(frame_seq), widgets = widgets).start()

	vid_frame = np.zeros(shape = frame_shape, dtype = np.uint8)

	for i, frame in enumerate(frame_seq):
		shape = frame.shape
		vid_frame.fill(0)
		vid_frame[:shape[0],:shape[1],:] = frame[:,:,:]
		video.write(vid_frame)
		bar.update(i)
	video.release()
	bar.finish()

	toc = time.perf_counter()
	print(f"{filepath} saved in {toc-tic:0.4f} seconds!\n")

def sorted_alphanumeric(data):
	'''
	Sort a list of alphanumeric string properly.

		Arguments:
			data(list of str):
				the list of str to be sorted
		
		Returns:
			data(list of str):
				the data in alphanumeric order

	'''
	convert = lambda text: int(text) if text.isdigit() else text.lower()
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(data, key=alphanum_key)

def plt_config(figsize=(16,9), dpi=300, linewidth=2, fontsize = 16, transparent=True, legend_loc='upper left'):
	'''
	Configure visual aspects of figures by changing default rc parameters.
	Arguments and hardcoded parameters can be increased if required.
	Detailed information about available rc parameters here :
	https://matplotlib.org/3.3.3/tutorials/introductory/customizing.html#a-sample-matplotlibrc-file
	
		Arguments:
			figsize(int tuple): 
				figure size in inches
				Default : (16,9)

			dpi(int): 
				dots(pixels) per inch. The more dpi chosen the more resolution in the image
				Default : 300

			linewidth(int):
				linewidth of the lines to be plotted
				Default : 2

			fontsize(int):
				fontsize for labels, ticks and anything else on the figure
				Default : 16

			transparent(bool):
				determines if the background will be transparent or not
				Default : True

	'''
	lines = {'linewidth' : linewidth}

	font = {'family' : 'sans-serif',
			'style'  : 'normal',
			'weight' : 'normal',
			'size'   : fontsize}

	axes = {'facecolor'		: 'white',  	# axes background color
			'edgecolor'		: 'black',  	# axes edge color
			'linewidth'		: 1,  			# edge linewidth
			'grid'			: True,   		# display grid or not
			'titlesize'		: 'large',  
			'titleweight'	: 'bold',  
			'titlepad'		: 6.0,      	# pad between axes and title in points
			'labelsize'		: 'medium',
			'labelpad'		: 7.0,     		# space between label and axis
			'labelweight'	: 'bold',  		# weight of the x and y labels
			'xmargin'		: .05,
			'ymargin'		: .05}
	
	xtick = {'major.pad' : 10.0}

	ytick = {'major.pad' : 10.0}
	
	grid = {'color'			: 'b0b0b0',  	# grid color
			'linestyle'		: 'dashed',  	# solid
			'linewidth'		: .8,     	 	# in points
			'alpha'			: .5}     	 	# transparency, between 0.0 and 1.0

	legend = {'loc'			: legend_loc,
			'frameon'		: 'True',  		# if True, draw the legend on a background patch
			'framealpha'	: 0.8,     		# legend patch transparency
			'fancybox'		: False,   		# if True, use a rounded box for the legend background, else a rectangle
			'fontsize'		: 'medium'}

	figure = {'figsize'	: figsize,  		# figure size in inches
			  'dpi'		: dpi}      		# figure dots per inch

	savefig = {'dpi'		: dpi,       	# figure dots per inch or 'figure'
			   'format'		: 'png',
			   'bbox'		: 'tight',   	# {tight, standard}
			   'pad_inches'	: .3,    		# Padding to be used when bbox is set to 'tight'
			   'transparent': transparent}
	
	config = {'lines' 	: lines, 
			  'font' 	: font, 
			  'axes' 	: axes, 
			  'xtick' 	: xtick,
			  'ytick'	: ytick,
			  'grid' 	: grid, 
			  'legend' 	: legend, 
			  'figure' 	: figure, 
			  'savefig' : savefig}

	for key, val in config.items():
		rc(key, **val)

# ----------- Seciton 4 : Processing ----------- # 

def process_image(in_file, processor):
	'''
	Process an image file and overlay pupil&iris

		Arguments:
			in_file(str):
				input file path

		Return:
			frame(np.ndarray of dim=3)
				frame with pupil overlay
	'''
	frame = cv2.imread(in_file)
	frame = processor(frame)
	return frame

def process_video(in_file, processor, percentage=1.0):
	'''
	Process a video file and overlay pupil&iris

		Arguments:
			in_file(str):
				input file path

		Return:
			frame_seq(list of np.ndarrays of dim = 3):
				list of frames to be saved as a video
	'''
	# Init
	tic = time.perf_counter()
	video = cv2.VideoCapture(in_file)
	fps = video.get(cv2.CAP_PROP_FPS)
	frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
	video_pupil = [None]*int(frame_count*percentage)
	
	# The percentage representation
	print(f"{in_file} processing...")
	widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
	bar = progressbar.ProgressBar(maxval=int(len(video_pupil)), widgets = widgets).start()

	# Iterate through the frames in the video
	for i,frame in enumerate(video_pupil):
		ret, frame = video.read()
		if not ret:
			break
		video_pupil[i] = processor(frame)
		# print(processor.last_pupil)
		bar.update(i)

	bar.finish()
	toc = time.perf_counter()
	print(f"{in_file} processed in {toc-tic:0.4f} seconds!\n")

	return video_pupil, fps

class PupirisOnFrame():
	'''
	Find the pupil and iris on frame and overlay

	STEP 1 : Detect the region of interest for iris and pupil
	STEP 2 : Find adaptive iris and pupil thresholds
	STEP 3 : Find exact locations of the iris and the pupil
	STEP 4 : Check Iris and Pupil
	STEP 5 : Record the status
	STEP 6 : Overlay rectangles, iris, and pupil
	'''
	def __init__(self, update_limit, radius_range, 
				 r_step=8, xy_step=8, np_clusters = 2, ni_clusters = 3,
				 h_factor = 1.2, i_factor=2.5, response_margin=.2, 
				 window_size=5, rect_pupil=True, iris=True, rect_iris = True,
				 pupil_color=(0,255,0), iris_color=(255,255,0)):
		'''
		Data segment initialization
			Arguments:
				update_limit(tuple of int):
					Update limits for recalculating haar radius
					(radius_update_limit, threshold_update_limit)

				radius_range(tuple of int):
					range to be swapped during haar-radius search

				r_step(int):
					radius search increment
					Default = 8

				xy_step(tuple of int/int):
					x and y increment in the convolution operation of
					radius search 
					Default = 8

				 np_clusters(int):
				 	number of clusters for color segmentation of pupil
				 	Default = 3

				 ni_clusters(int):
				 	number of cluster for color segmentation of iris
				 	Default = 6

				 h_factor(float):
				 	rectangle enlargement factor for best haar-radius  
				 	found in radius search 
				 	Default = 1.4

				 i_factor(float):
				 	rectangle enlargement factor for 
				 	obtaining eye region out of best pupil region
				 	Default = 1.8

				 response_margin(float):
				 	Allowed percent deviation from average response 
				 	coming from haar-surround feature detector
				 	Used in blink detection algorithm
				 	Default = .2

				 window_size(int):
					Allowed storage amount for response record
				 	Default = 5
				 	
				 rect_pupil(bool):
					overlay region of pupil detection or not
				 	Default=True

				 iris(bool):
				 	Overlay iris or not
				 	Default=True

				 rect_iris(bool):
					overlay region of iris detection or not
				 	Default = True

				 pupil_color(tuple of int):
				 	overlay color for pupil in BGR 
				 	Default = (0,255,0)

				 iris_color(tuple of int):
				 	overlay color for iris in BGR
				 	Default = (255,255,0)
		'''
		# User defined
		self.radius_update_limit = update_limit[0]
		self.threshold_update_limit = update_limit[1]
		self.radius_range = radius_range
		
		# With a default value
		self.r_step = r_step
		self.xy_step = xy_step
		self.np_clusters = np_clusters
		self.ni_clusters = ni_clusters
		self.h_factor = h_factor
		self.i_factor = i_factor
		self.response_margin = response_margin
		self.window_size = window_size
		self.rect_pupil = rect_pupil
		self.iris_flag = iris
		self.rect_iris = rect_iris
		self.pupil_color = pupil_color
		self.iris_color = iris_color

		# Internal Structure
		self.response_rec = []
		self.radius_counter = 0
		self.threshold_counter = 0
		self.last_pupil = None
		self.last_iris = None
		self.R3 = None # pupil_region, response, radius
		self.p_th = None
		self.i_th = None
		self.blink_flag = False
		self.pupil_roi = None
		self.eye_region = None
		self.eye_roi = None

	def __call__(self, frame):
		'''
		Core processor for a frame. All the main operations are called here
			
			Arguments:
				frame(np.ndarray of dim = 2 or 3):
					the frame to processed

			Returns:
				frame(np.ndarray of dim = 2 or 3):
					the frame with segmented pupil and iris 
		'''

		# STEP 1 : Detect the region of interest for iris and pupil
		if self.radius_update():
			
			pupil_region,response,radius  = radius_search(frame, self.radius_range, r_step=self.r_step, xy_step=self.xy_step, h_factor=self.h_factor)
			if response  < self.blink_th():
				# Record the response
				self.R3 = pupil_region,response,radius
				self.blink_flag = False
				self.push_rec(response)
				self.pupil_roi = rect_clip(frame, pupil_region)

				if self.iris_flag:
					self.eye_region = rect_enlarge(pupil_region, self.i_factor, frame.shape)	
					self.eye_roi = rect_clip(frame,self.eye_region)

			else:
				self.blink_flag = True

		# STEP 2 : Find adaptive iris and pupil thresholds
		if self.threshold_update():
			self.p_th= get_threshold(self.pupil_roi, n_clusters = self.np_clusters)

			# Bigger roi for iris
			if self.iris_flag:
				self.i_th= get_threshold(self.eye_roi, n_clusters = self.ni_clusters)
	
		# Check blink status
		if not self.blink_flag:
			pupil_region,response,radius = self.R3
			iris = None

			# STEP 3 : Find exact locations of the iris and the pupil
			th_pupil = self.p_th[0]

			pupil = get_pupiris(self.pupil_roi, th_pupil)

			if self.iris_flag:
				th_iris = self.i_th[0]
				iris = get_pupiris(self.eye_roi, th_iris)

			# STEP 4 : Check Iris and Pupil
			pupil, iris = self.pupiris_check(pupil,iris,pupil_region,self.eye_region)
			
			# STEP 5 : Record the status
			status = {'haar_radius' : radius,
					  'pupil_region': pupil_region,
					  'pupil threshold' : self.p_th}

			# STEP 6 : Overlay search regions, iris, and pupil
			if self.rect_pupil:
				# print(frame.shape)
				# print(self.pupil_color)
				frame = overlay_rectangle(frame, pupil_region, color=self.pupil_color, thickness=1)
			
			if self.iris_flag:
				if self.rect_iris:
					frame = overlay_rectangle(frame, self.eye_region, color=self.iris_color, thickness=1)
				# Status
				status['eye_region'] = self.eye_region
				status['iris_threshold'] = self.i_th

			if pupil is not None:
				frame = overlay_ellipse(frame, pupil, color = self.pupil_color)
				# Status
				status['pupiris_center'] = pupil[0]
				status['pupiris_theta'] = pupil[2]
				status['pupil_ax'] = pupil[1]
				
			
			if self.iris_flag and iris is not None:
				frame = overlay_ellipse(frame, iris, color = self.iris_color, center = False)
				status['iris_ax'] = iris[1]
		else:
			status = {'BLINK' : 1}

		frame = overlay_status(frame, status)
		return frame

	def radius_update(self):
		'''
		Do the radius search again or not

			Returns:
				flag(bool):
					Boolean flag for radius search decision
		'''
		self.radius_counter, flag = update_counter(self.radius_update_limit, self.radius_counter)
		return flag

	def threshold_update(self):
		'''
		Recalculate the adaptive color threshold or not

			Returns:
				flag(bool):
					Boolean flag for recalculate threshold decision
		'''
		self.threshold_counter, flag = update_counter(self.threshold_update_limit, self.threshold_counter)
		return flag

	def blink_th(self):
		'''
		Calculates the blink threshold averaging the response record

			Returns:
				threshold(float):
					the response threshold for blink decision
		'''
		if self.response_rec:
			av_response = np.mean(self.response_rec)
			threshold = av_response*(1-self.response_margin)
		else:
			threshold = float('inf')
		return threshold

	def push_rec(self, response):
		'''
		Push a response to the response record
		Delete the oldest record if window size has passed

			Arguments:
				reponse(float):
					response to be stored
		'''
		if len(self.response_rec)<self.window_size:
			self.response_rec.append(response)

		else:
			self.response_rec = self.response_rec[1:] + [response]

	def pupiris_check(self, pupil, iris, pupil_region, eye_region):
		'''
		Check pupil and iris if they fit in their regions and
		offset them to overlay on the frame

			Arguments:
				pupil(tuple of tuples of float):
					Pupil ellipse parameters
					((x_center, y_center), (major_ax,minor_ax), theta)

				iris(tuple of tuples of float):
					Iris ellipse parameters
					((x_center, y_center), (major_ax,minor_ax), theta)

				pupil_region(tuple of tuple of int):
					rectangle involving pupil
					opencv type rectangle definition
					(x_start, y_start), (x_end, y_end)

				eye_region(tuple of tuple of int):
					rectangle involving iris and pupil
					opencv type rectangle definition
					(x_start, y_start), (x_end, y_end)
		'''
		if ellipse_check(pupil,pupil_region):
			pupil= ellipse_offset(pupil,pupil_region[0])
			self.last_pupil = pupil

		else:			
			pupil = self.last_pupil

		if ellipse_check(iris,eye_region):
			iris = ellipse_offset(iris,eye_region[0])
			iris = ratio_check(pupil, iris)
			self.last_iris = iris

		else:
			iris = self.last_iris

		# if iris is not None and pupil is not None:

		return pupil, iris

## ----------- Section 4.1 : Process Control ----------- ##

def ratio_check(pupil,iris,ratio=2, minor_major=1.2):
	'''
	Iris detection is untrustable.
	Check the ratio depending on the pupil.
	Correct the center, angle and axis ratio if necessary

		Arguments:
			pupil(tuple of tuples of float):
				Pupil ellipse parameters
				((x_center, y_center), (major_ax,minor_ax), theta)

			iris(tuple of tuples of float):
				Iris ellipse parameters
				((x_center, y_center), (major_ax,minor_ax), theta)

			ratio(float):
				minimum ratio of minor axis of pupil and iris

			minor_major(float):
				minimum ratio of minor and major axis of iris

		Returns:
			iris(tuple of tuples of float):
				Corrected iris ellipse parameters
				((x_center, y_center), (major_ax,minor_ax), theta)
	'''

	# Equi-center:

	ma_iris, MA_iris = iris[1]
	ma_pupil, MA_pupil = pupil[1]

	if(ma_iris < ma_pupil*ratio):
		ma_iris = ma_pupil*ratio

	ratio = ratio*minor_major
	if (MA_iris < MA_pupil*ratio):
		MA_iris = MA_pupil*ratio

	iris = (pupil[0], (ma_iris, MA_iris), pupil[2])

	return iris

def ellipse_check(ellipse, rectangle):
	'''
	Return false if ellipse does not fit rectangle
	or ellipse is None

		Arguments:
			ellipse(tuple of tuples of float):
				Ellipse parameters
				((x_center, y_center), (major_ax,minor_ax), theta)

			rectangle(tuple of tuple of int):
				rectangle candidate involving ellipse
				opencv type rectangle definition
				(x_start, y_start), (x_end, y_end)

		Returns:
			flag(bool):
				true if ellipse is inside the rectangle
	'''
	flag = False

	if ellipse is not None:
		flag = ellipse_in_region(ellipse, rectangle)

	return flag

def ellipse_in_region(ellipse, rectangle, up=1, low=0.1):
	'''
	STEP 1 : Check if the center is inside the rectangle 
	STEP 2 : Check if the major and minor axes of the ellipse 
	are greater than <low> percent of the max(width, heigth)
	STEP 3 : Check if the major and minor axes of the ellipse 
	are less than <up> percent of the max(width, heigth)

		Arguments:
			ellipse(tuple of tuples of float):
				Ellipse parameters
				((x_center, y_center), (major_ax,minor_ax), theta)

			rectangle(tuple of tuple of int):
				rectangle candidate involving ellipse
				opencv type rectangle definition
				(x_start, y_start), (x_end, y_end)

			up(float):
				The percentage of major or minor axis has to be
				less than the max(width, heigth)

			low(float):
				The percentage of major or minor axis has to be
				greater than the max(width, heigth)

		Returns:
			flag(bool):
				true if ellipse is inside the rectangle
	'''
	(x, y), (MA, ma), angle = ellipse

	width = rectangle[1][0] - rectangle[0][0]
	heigth = rectangle[1][1] - rectangle[0][1]

	if x<0 or x>width:
		return False

	if y<0 or y>heigth:
		return False

	# Center is in the region but what about the width and height?
	
	hyp = max(width, heigth)
	upper = hyp*up
	lower = hyp*low

	if MA > upper or MA < lower:
		return False

	if ma > upper or ma < lower:
		return False

	return True

def update_counter(update_lim, counter):
	'''
	Update the counter and check if it reached to the limit

		Arguments:
			update_lim(int):
				the upper limit for counting

			counter(int):
				the last state of the counter

		Returns:
			counter(int):
				the next state of the counter

			flag(bool):
				the flag for counter reached to the limit or not
	'''

	flag = False

	if counter == 0:
		flag =True

	elif counter>=update_lim:
		counter = 0
		flag = True

	counter+=1;
	return counter, flag

# ----------- Section 5 : Haar Like Feature Extraction ----------- # 

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

			Returns:
				kernel(np.ndarray of dim=2):
					Haar Surround Kernel in dense format
		'''
		kernel = np.zeros(shape=(2*self.r_out-1, 2*self.r_out-1), dtype=np.float32)
		kernel.fill(self.val_out)

		start = (self.r_out-self.r_in)
		end = (self.r_out+self.r_in-1)

		kernel[start:end, start:end] = self.val_in
		return kernel

def detect_pupil(frame, radius, step, h_factor=1.4):
	'''
	!!! ASSUMPTION : Pupil can be roughly described as
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
	gray_frame = to_gray(frame)
	row, col = gray_frame.shape
	pad  = 2*radius
	haar_radius = int(h_factor*radius)
	
	# STEP 1 : Get the integral image
	# Need to pad by an additional 1 to get bottom & right edges.
	frame_pad = cv2.copyMakeBorder(gray_frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE) 
	frame_int = cv2.integral(frame_pad)

	# STEP 2 : Convolution
	surround = HaarSurroundFeature(radius)
	filtered, response, center = conv_int(frame_int, surround, step, pad)
	region = get_roi(frame, center, haar_radius)

	return region, response, filtered

def radius_search(frame, radius, r_step, xy_step=8, h_factor=1.4):
	'''
	Linear search for the best radius, which returns the smallest response

		Arguments:
			radius(tuple of int):
				range to be swapped during haar-radius search

			frame(np.ndarray of dim = 2 or 3):
				the frame to processed

			radius(int):
				inner radius for the HaarSurroundFeature kernel

			step(tuple of int/int):
				ystep, xstep
				stepsize for kernel walk

			h_factor(float):
				factor of increase in calculating 
				haar_radius = h_factor*radius
				Default=1.5	

			Returns:
				best_region(tuple of tuple of int):
					best region estimation in the form
					of a opencv rectangle 
					((x_start, y_start), (x_end, y_end))

				best_response(float):
					the smallest response obtained in the search range

				best_radius(int): 
					inner radius of the haar kernel for which the best response 
	'''
	best_region = 0
	best_response = 0
	best_radius = 0

	if not isinstance(radius, (tuple, list)):
		if isinstance(radius, int):
			radius = (radius, radius+1)
		else:
			info = 'Unsupported radius respresentation\n'
			info += 'Example : radius = (2,8)'
			raise ValueError(info)

	for radius in range(*radius,r_step):
		region, response, _ = detect_pupil(frame, radius=radius, step=xy_step, h_factor=h_factor)
		if response < best_response:
			best_radius = radius
			best_response = response
			best_region = region

	return best_region, best_response, best_radius


## ----------- Section 5.1 : Integral Convolution ----------- ##

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
	return frame_conv, min_response, center

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

# ----------- Section 6 : Pupil/Iris Segmentation ----------- # 

def get_pupiris(frame_roi, threshold):
	'''
	Get the pupil or iris using an adaptive threshold (from color segmentation)
	Extract the candidate contours from the binary thresholded image
	Fit an ellipse to the best candidate	

		Arguments:
			frame_roi(np.ndarray of dim = 2):
				grayscale region of interest which harbor the pupil

			threshold(int):
				the pixel threshold obtained in the color segmentation phase
		
		Returns:
			pupiris_ellipse(tuple of tuples of float):
				the parameters of the ellipse overlaying the pupil/iris
				((x_center, y_center), (major_ax,minor_ax), theta)
	'''

	pupiris = None
	best_ellipse = ((0,0),(0,0),0)
	contours = get_contours(frame_roi, threshold)

	if len(contours):
		for contour in contours:
			candidate = fit_ellipse_LSQ(contour)
			if np.sum(candidate[1]) > np.sum(best_ellipse[1]):
				best_ellipse = candidate
				
			pupiris = best_ellipse
	return pupiris

## ----------- Section 6.1 : Color Segmentation ----------- ##

def get_threshold(frame, n_clusters=3):
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
			
			n_clusters(int):
				number of clusters to heap the pixels together
				Default = 3

		Returns:
			threshold(int):
				adaptive threshold pixel value
	'''

	flat_frame = frame.reshape(-1, 1)
	kmeans = KMeans(n_clusters=n_clusters, random_state=0)

	kmeans.fit(flat_frame)
	labels = kmeans.labels_
	centers = kmeans.cluster_centers_.flatten()

	thresholds = [flat_frame[labels==cid].max() for cid in range(len(centers))]

	return sorted(thresholds)

def clustered_frame(frame, n_clusters):
	'''
	Creates a color segmented frame using KMeans clustering
	For visual debugging

		Arguments:
			frame(np.ndarray of dim = 2 or 3):
				the frame of interest
			
			n_clusters(int):
				number of clusters to heap the pixels together
				Default = 3

		Returns:
			frame_clustered(np.ndarray of dim = 2 or 3):
				the color segmented
	'''
	flat_frame = frame.reshape(-1, 1)
	kmeans = KMeans(n_clusters=n_clusters, random_state=0)
	kmeans.fit(flat_frame)
	labels = kmeans.labels_

	px_val = [px for px in range(0,256,255//(n_clusters-1))]
	assert len(px_val) == n_clusters
	order = np.argsort(kmeans.cluster_centers_.flatten())
	px_val = np.asarray(px_val, dtype=np.uint8)

	# Arrange the pixel values such that the pupil 
	# will have the darkest region
	px_val[order]=np.sort(px_val)

	data = px_val[labels.flatten()] 
	frame_clustered = data.reshape((frame.shape)) 

	return frame_clustered

def kmeans_elbow(frame, search=(1,16)):
	'''
	In cluster analysis, the elbow method is a heuristic 
	used in determining the number of clusters in a data set. 
	The method consists of plotting the explained variation 
	as a function of the number of clusters, and picking 
	the elbow of the curve as the number of clusters to use.

	Shows the percent of variance explained plot to help to
	find the right number of clusters

		Arguments:
			frame(np.ndarray of dim = 2 or 3):
				the frame to be segmented

			search(tuple of int):
				the range for cluster search
	'''
	kmeans = KMeans(random_state=0)
	flat_frame = frame.reshape(-1, 1)
	plt.figure()
	visualizer = KElbowVisualizer(kmeans, k=search)
	visualizer.fit(flat_frame)        
	visualizer.show()
	plt.close()

## ----------- Section 6.2 : Ellipse Fitting ----------- ##

def get_contours(frame_roi, th, th_max=255, kernel=(5,5)):
	'''
	Extract the candidate contours which are to be 
	used in ellipse fitting, from a binary thresholded image

		Arguments:
			frame_roi(np.ndarray of dim = 2):
				grayscale region of interest which harbor the pupil
			
			th(int):
				pixel threshold value
			
			th_max(int):
				maximum pixel value to use with the THRESH_BINARY
			
			gaussion_kernel(tuple of int):
				kernel size of the morphological open filter
				Default=(5,5)

		Returns:
			contours(list of dim = 4):
				list of contours found using canny edge detection on
				binary thresholded image.
				sorted by area.
	'''

	threshold = cv2.morphologyEx(frame_roi, cv2.MORPH_OPEN, kernel)
	_, threshold = cv2.threshold(threshold, th, th_max, cv2.THRESH_BINARY_INV)

	# threshold = cv2.blur(threshold,kernel)
	# for i in range(blur_iter):
	# 	threshold = cv2.GaussianBlur(threshold,kernel,0)
	

	edges = cv2.Canny(threshold,0,255)

	# cv2.imshow("threshold", edges)
	# key = cv2.waitKey(0)

	contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
	# contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
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

# ----------- Section 7 : Main Operation ----------- #

def main(mode, in_file, out_file, percentage=1.0):
	'''
	Tracks the pupil and the iris of a human eye. 
	Analyze a video feed or an image and highlight 
	the centers and perimeters of the pupil and the iris.
	
	(Live mode is under construction)

		Arguments:
			mode(str):
				operating mode of the system
				image, video or live

			in_file(str):
				input filepath

			out_file(str):
				output filepath

			percentage(float):
				the percentage of video to be processed
	'''
	if mode == 'image':
		pof = PupirisOnFrame(update_limit=(1,4), radius_range=(48,96), r_step=8, xy_step=8, iris=True)
		image_pupil = process_image(in_file, pof)
		cv2.imwrite(out_file, image_pupil)

	elif mode == 'video':
		pof = PupirisOnFrame(update_limit=(1,8), radius_range=(48,64), r_step=4, xy_step=8, np_clusters=2, ni_clusters=3, h_factor=1.2, iris=True)
		video_pupil, fps = process_video(in_file, pof, percentage)
		save_video(video_pupil, out_file, fps)

	elif mode == 'live':
		None

	else:
		info = 'Unsupported mode of operation!\n'
		info += 'Refer to user manual by eye_track -h'
		raise ValueError(info)

def parse_argv(argv):
	'''
	Parse the argument vector
	Example:
	python3 eye_track.py -m video -i eye_recording.flv -o eye_rec_pupil.avi

		Arguments:
			argv(list of str):
				command line arguments passed to the main function
				(look at the user manual for accepted parameters)

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
	info += "i.e. image/video/live\n\n"
	info += "-i <input file>\n"
	info += "i.e. eye_rec.avi\n\n"
	info += "-o <output file>\n"
	info += "i.e. eye_rec_pupil.avi\n\n"
	info += "-p <percentage> : the percentage of video to be processed\n"
	info += "i.e. 0.2\n\n"

	return info

if __name__ == '__main__':

	if len(sys.argv)>1:
		mode, in_file, out_file, percentage = parse_argv(sys.argv)
		if mode is not None:
			main(mode, in_file, out_file, percentage)
	else:
		print(user_manual())
