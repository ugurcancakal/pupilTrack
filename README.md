# EyeTrack

Implementation of a system that tracks the pupil and the iris of a human eye. System will analyze a video feed and highlight centers and perimeters of the pupil and the iris.


## Environment:

| Feature       | Description 			|
| -----------   | ----------- 			|
| CPU           | Intel Core i7 - 7500 	|
| CPU Speed     | 2.7 GHz - 2.9 GHz     |
| RAM 			| 8 GB					|
| OS			| Ubuntu 20.04			|
| System 		| 64 - bit 				|
| Programming	| Python 3 				|

## Requirements:

	matplotlib==3.2.2
	numpy==1.18.5
	scipy==1.5.0
	yellowbrick==1.2.1
	progressbar33==2.4
	scikit_learn==0.23.2
	opencv-python==4.4.0.46

To install all:

	pip3 install -r requirements.txt

## Example Usage: 

*Command Prompt:*

	python3 eye_track.py -m video -i ./test_input/eye_left.avi -o ./test_output/eye_left_processed.avi -p .1

### Command Line Arguments:

| Argument  | Meaning 						 | Example 								|
| --------- | ------------------------------ | ------------------------------------ |
| -m 	    | operation mode 				 | image/video 							|
| -i 	    | input filepath 				 | ./test_input/eye_left.avi 			|
| -o 		| output filepath 				 | ./test_output/eye_left_processed.avi |
| -p 		| the percentage to be processed | 0.1 									|

Average processing time for a **39 seconds(24 fps)** grayscale video 
with dimensions 620x420 is **300 seconds** on average in configuration:

| Parameter  		   	|
| ------------------- 	|
| update_limit=(1,8)   	|
| radius_range=(48,64) 	|
| r_step=4 				|
| xy_step=8 			| 
| np_clusters=2 		|
| ni_clusters=3 		|

*all the rest : default parameters*

## Disclaimer

Unittests are implemented for leading the implementation process. They can be used to observe and test the middle steps. However, they are in progress for now and does not represent a proper test environment. 

