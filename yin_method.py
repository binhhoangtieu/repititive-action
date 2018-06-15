import numpy as np
import cv2
import matplotlib.pyplot as plt
# filename = './person18_boxing_d3_uncomp.avi'
# filename = './jump.mp4'
# filename = './jump_crop.mp4'
filename = './Clock_Lunge_cut.mp4'

""" YIN method:
Based on the paper Automatic Detection of Repetitive Actions in a VideoCapture - Hassan WEHBE
YIN algorithm

1. Represent images waveform signal
2. Calculate YIN matrix
3. Extract the triangles
4. Deduce repetition details

"""


def main():
	frame_array = read_video_from_file(filename)
	YIN_signal = frame_to_signal(frame_array)

	#Create YIN_matrix
	YIN_matrix = YIN_signal_to_matrix(YIN_signal)

	# print(YIN_matrix.shape)
	
	t = np.linspace(0, len(YIN_signal), len(YIN_signal))
	gradient_matrix = np.sum(np.array(np.gradient(YIN_matrix)),axis=0)



	plt.plot(t,YIN_signal)
	plt.show()

	# print(gradient_matrix)

	plt.imshow(YIN_matrix)
	plt.show()


	# Vote_matrix = right_angle_vote(YIN_matrix,gradient_matrix)
	# plt.imshow(Vote_matrix)
	# plt.show()
	# print_matrix_to_file(YIN_matrix,'Yin.csv')
	# print_matrix_to_file(Vote_matrix,'Vote.csv')
	

def print_matrix_to_file(mat,file='./outfile.csv'):

	with open(file,'w+') as f:
		# for line in mat:
		# 	np.savetxt(f, line, fmt='%.2f', delimiter=' ')
		np.savetxt(f, mat, fmt='%.2f', delimiter=' ')

def repetitive_detect(yin):
	return

def right_angle_vote(Y, G):
	len_signal = len(Y)
	V = np.zeros((len_signal,len_signal,len_signal))
	Vote = np.zeros((len_signal,len_signal))
	for i in range(len_signal):
		for j in range(i+1, len_signal):
			S1 = 0
			S4 = 0
			for t in range(j+1, len_signal-j):
				for a in range(t):
					for b in range(t-a):
						S1 += Y[i+a,j+b]/(t+1)*(t-a+1)
				S2=0
				S3 =0
				S4 = 0
				for k in range(j+1,j+t):
					S2 += G[i,k]
				for k in range(i+1,i+t):
					S3 += G[k,j]
				S4 += (S2+S3)/(2*t)
				V[i,j,t] = S1*S4
			Vote[i,j] = max(V[i,j,:])
			print(i,j,Vote[i,j])
	print(Vote.shape)
	return(Vote)

def YIN_signal_to_matrix(signal):
	len_signal = len(signal)
	YIN_matrix = np.zeros((len_signal,len_signal),dtype=float)
	for j in range(len_signal):
		for i in range(1,len_signal-j):
			YIN_matrix[i,j] = min(signal[j:j+i]) # change position of i, j is important
	return YIN_matrix


def YIN_signal_to_matrix_old(signal):
	len_signal = len(signal)
	YIN_matrix = np.zeros((len_signal,len_signal),dtype=float)
	for i in range(len_signal):
		for j in range(i,len_signal):
			YIN_matrix[i,j] = min(signal[j:j+i+1]) # change position of i, j is important
	return YIN_matrix


def frame_to_signal(frame_array):
	array_size = len(frame_array)
	signal = np.zeros(array_size)
	for i in range(len(frame_array)):
		signal[i] = np.sum(frame_array[i])
	# Normalization
	signal=cv2.normalize(signal,signal,0,255,cv2.NORM_MINMAX)
	# signal=cv2.normalize(signal,signal,0,1,cv2.NORM_MINMAX,dtype=cv2.CV_32F)
	# return signal[0:100]
	return signal

def read_video_from_file(filename):
	video_cap = cv2.VideoCapture(filename)
	nframe = np.int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frameWidth = np.int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frameHeight = np.int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))	
	j = 0
	ret = True
	cap_images = np.empty((nframe, frameHeight, frameWidth, 3))
	while (j < nframe  and ret):
		ret, cap_images[j] = video_cap.read()
		if ret != True: 
			cap_images = cap_images[0:j-1]
			break
		else:
			j += 1
	return cap_images

def normalization(frame_array):
	MAX=max(frame_array)
	MIN=min(frame_array)
	MEAN=np.mean(frame_array)
	return (frame_array - MIN)/(MAX-MIN)

if __name__ == '__main__':
	main()
