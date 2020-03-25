import numpy as np
import cv2
from rotations import *
from matplotlib import pyplot as plt
STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500

lk_params = dict(winSize  = (21, 21),
				#maxLevel = 3,
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def featureTracking(image_ref, image_cur, px_ref):
	kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]

	st = st.reshape(st.shape[0])
	kp1 = px_ref[st == 1]
	kp2 = kp2[st == 1]

	return kp1, kp2

def camera_matrices(R_old, R, t, K):
	P_0 = np.zeros((3, 4))
	tilt = 4.6
	init_R = rotateZ(tilt)[0:3, 0:3]
	P_0[0:3, 0:3] = init_R
	P_1 = np.zeros((3, 4))
	P_1[0:3, 0:3] = R
	P_1[:, 3] = np.squeeze(t)
	return K @ P_0, K @ P_1

class PinholeCamera:
	def __init__(self, width, height, fx, fy, cx, cy,
				k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.distortion = (abs(k1) > 0.0000001)
		self.d = [k1, k2, p1, p2, k3]


class VisualOdometry:
	def __init__(self, cam, annotations):
		self.frame_stage = 0
		self.cam = cam
		self.new_frame = None
		self.last_frame = None
		self.cur_R = None
		self.cur_t = None
		self.px_ref = None
		self.px_cur = None
		self.px_planar_ref = None
		self.px_planar_cur = None
		self.scale = 1
		self.focal = cam.fx
		self.pp = (cam.cx, cam.cy)
		self.k = np.array([[self.focal, 0, cam.cx], [0, self.focal, cam.cy], [0, 0, 1]])
		self.trueX, self.trueY, self.trueZ = 0, 0, 0
		self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
		with open(annotations) as f:
			self.annotations = f.readlines()

	def getAbsoluteScale(self, frame_id):  #specialized for KITTI odometry dataset (TAKEN FROM GROUND TRUTH)
		ss = self.annotations[frame_id-1].strip().split()
		x_prev = float(ss[3])
		y_prev = float(ss[7])
		z_prev = float(ss[11])
		ss = self.annotations[frame_id].strip().split()
		x = float(ss[3])
		y = float(ss[7])
		z = float(ss[11])
		self.trueX, self.trueY, self.trueZ = x, y, z
		return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))

	def processFirstFrame(self):
		self.px_ref = self.detector.detect(self.new_frame)
		self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
		self.frame_stage = STAGE_SECOND_FRAME

	def processSecondFrame(self):
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		self.frame_stage = STAGE_DEFAULT_FRAME
		self.px_ref = self.px_cur

	def processFrame(self, frame_id):
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		absolute_scale = self.getAbsoluteScale(frame_id)


		## Changes made from here ##
		self.track_window(self.last_frame, self.new_frame)
		P0, P1 = camera_matrices(self.cur_R, R, t, self.k)
		points = cv2.triangulatePoints(P0, P1, self.px_planar_ref.T, self.px_planar_cur.T)
		points = points.T
		reproj_pix = np.zeros_like(points)
		for i in range(len(points)):
			reproj_pix[i,0:3] = -P1 @ points[i]
			points[i] /= points[i,3]
			print(points[i])
		y_mean = np.mean(abs(points[:,1]))
		scale = 1.65/y_mean
		alpha = 0.5
		offset = 0
		if(abs(np.amax(abs(points[:,1]) - y_mean))) > 0.5 or y_mean < 1:
			self.scale = self.scale
		else:
			self.scale = (1-alpha)*self.scale + alpha*scale + offset
		print(absolute_scale)
		print(self.scale)
		print('')
		#print(reproj_pix)
		#print(self.px_planar_cur)

		## TO HERE ##
		
		if(absolute_scale > 0.1):
			self.cur_t = self.cur_t + self.scale*self.cur_R.dot(t)
			self.cur_R = R.dot(self.cur_R)
		if(self.px_ref.shape[0] < kMinNumFeature):
			self.px_cur = self.detector.detect(self.new_frame)
			self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
		self.px_ref = self.px_cur

	def update(self, img, frame_id):
		assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		self.new_frame = img
		if(self.frame_stage == STAGE_DEFAULT_FRAME):
			self.processFrame(frame_id)
		elif(self.frame_stage == STAGE_SECOND_FRAME):
			self.processSecondFrame()
		elif(self.frame_stage == STAGE_FIRST_FRAME):
			self.processFirstFrame()
		self.last_frame = self.new_frame

	def track_window(self, im1, im2):
		"""Made as part of EiT computer vision village project"""
		FEATURE_PARAMS = dict(maxCorners=10,qualityLevel=0.001,minDistance=20,blockSize=7)
		LK_PARAMS = dict(winSize=(21, 21), maxLevel = 3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

		crop_x_lower, crop_x_upper = 250, 320
		crop_y_lower, crop_y_upper = 520, 580
		crop_img1 = im1[crop_x_lower:crop_x_upper, crop_y_lower:crop_y_upper]

		p0 = cv2.goodFeaturesToTrack(crop_img1, **FEATURE_PARAMS)
		p0 = np.squeeze(p0)
		for i in p0:
			i[0] += crop_y_lower
			i[1] += crop_x_lower
		# Calculate optical flow
		p1, st, err = cv2.calcOpticalFlowPyrLK(im1, im2, p0, None, **LK_PARAMS)
		print('amount of tracked points are :', len(p1))
		for i in p1:
			i[0] = round(i[0])
			i[1] = round(i[1])
			self.px_planar_ref, self.px_planar_cur = np.squeeze(p0), np.squeeze(p1)

	def plot_correspondences(self):
		plt.subplot(2, 1, 1)
		plt.imshow(self.last_frame)
		plt.scatter(self.px_planar_ref[:, 0], self.px_planar_ref[:, 1], c='red')
		plt.subplot(2, 1, 2)
		plt.imshow(self.new_frame)
		plt.scatter(self.px_planar_cur[:, 0], self.px_planar_cur[:, 1], c='red')
		plt.show()