import numpy as np
import cv2
from rotations import *
import matplotlib.pyplot as plt
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
	P_0[0:3, 0:3] = R_old
	P_1 = np.zeros((3, 4))
	P_1[0:3, 0:3] = R @ R_old
	P_1[:, 3] = np.squeeze(t)
	return K @ P_0, K @ P_1


def plot_3d(points):
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.scatter3D(points[:,0], points[:, 1], points[:, 2])

def print_points(points):
	for i in points:
		print(i)
	print('\n')


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
		self.cur_t_unscaled = None
		self.px_ref = None
		self.px_cur = None
		self.px_planar_ref = None
		self.px_planar_cur = None
		self.scale = 1
		self.error = []
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

	def refine_points(self, F):
		n = len(self.px_planar_ref)
		p1, p2 = cv2.correctMatches(F,
			np.reshape(self.px_planar_ref, (1, n, 2)), np.reshape(self.px_planar_cur, (1, n, 2)))

		self.px_planar_ref, self.px_planar_cur = np.reshape(p1, (n,2)), np.reshape(p2, (n,2))


	def scale_dynamic(self, points, R):
		y_mean = np.mean(points[:, 1])
		scale = -1.65 / y_mean
		scale = abs(scale)
		alpha = 0.5
		rot_trace = abs(np.trace(R.dot(self.cur_R) - self.cur_R))
		if rot_trace > 5e-3 and self.scale > 0.4:
			self.scale -= 0.03

		elif (scale > 2 and rot_trace < 5e-3):
			self.scale = (1 - alpha) * self.scale + 1.3 * alpha
		if (scale >= self.scale):
			if (rot_trace < 5e-3):
					self.scale = (1 - alpha) * self.scale + alpha * min(self.scale + 0.1, scale)
		else:
			if (rot_trace < 5e-3):
				self.scale = (1 - alpha) * self.scale + alpha * max(self.scale - 0.1, scale)
		self.scale = np.clip(self.scale, 0.1, 3)

	def remove_outliers(self, R, t):
		self.track_window(self.last_frame, self.new_frame)
		P0, P1 = camera_matrices(self.cur_R, R, t, self.k)

		points = cv2.triangulatePoints(P0, P1, self.px_planar_ref.T, self.px_planar_cur.T)
		points = points.T


		reprojection_points = np.zeros((len(points), 3))
		homogenous_pts = np.zeros((len(points),4))
		for i in range(len(points)):
			homogenous_pts[i] = points[i]
			homogenous_pts[i] /= homogenous_pts[i, 3]
			reprojection_points[i] = -P1 @ points[i]
		reproj_error = abs(self.px_planar_cur - reprojection_points[:,:2])

		good_points = []
		medi = np.median(homogenous_pts[:,1])
		for i in range(len(reproj_error)):
			if(np.mean(reproj_error[i]) < 100 and homogenous_pts[i,1] < 0 and abs(homogenous_pts[i,1] - medi) < 2*self.scale):
				good_points.append(points[i])

		if len(good_points) == 0:
			index = np.argmin(np.mean(reproj_error, 0))
			good_points.append(points[index])
		count = 0
		good_pts_ary = np.zeros_like(good_points)
		for i in good_points:
			good_pts_ary[count] = i
			count += 1

		if(len(good_points) == 0):
			return []
		for i in range(len(good_points)):
			good_pts_ary[i] /= good_pts_ary[i, 3]
		return good_pts_ary

	def processFirstFrame(self):
		self.px_ref = self.detector.detect(self.new_frame)
		self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
		self.frame_stage = STAGE_SECOND_FRAME

	def processSecondFrame(self):
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		self.cur_t_unscaled = self.cur_t
		self.frame_stage = STAGE_DEFAULT_FRAME
		self.px_ref = self.px_cur

	def processFrame(self, frame_id):
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		absolute_scale = self.getAbsoluteScale(frame_id)


		## Our functions ##
		self.track_window(self.last_frame, self.new_frame)
		F, mask = cv2.findFundamentalMat(self.px_cur, self.px_ref, cv2.RANSAC)
		self.refine_points(F)
		points = self.remove_outliers(R, t)
		self.scale_dynamic(points, R)
		self.error.append(absolute_scale - self.scale)
		print(self.scale)
		print(absolute_scale)
		if(self.scale > 0.1):
			self.cur_t = self.cur_t + (self.scale)*self.cur_R.dot(t)
			self.cur_R = R.dot(self.cur_R)
			self.cur_t_unscaled = self.cur_t_unscaled + self.cur_R.dot(t)
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

	def track_window(self, im1, im2, x_lower = 275, x_upper = 360, y_lower = 450, y_upper = 780):
		"""Made as part of EiT computer vision village project"""
		FEATURE_PARAMS = dict(maxCorners=20, qualityLevel=0.005, minDistance=20, blockSize=7)
		LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
						 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
		crop_img1 = im2[x_lower:x_upper, y_lower:y_upper]
		p0 = cv2.goodFeaturesToTrack(crop_img1, **FEATURE_PARAMS)
		p0 = np.squeeze(p0)
		for i in p0:
			i[0] += y_lower
			i[1] += x_lower
		# Calculate optical flow
		p1, st, err = cv2.calcOpticalFlowPyrLK(im2, im1, p0, None, **LK_PARAMS)
		for i in p1:
			i[0] = round(i[0])
			i[1] = round(i[1])
			self.px_planar_cur, self.px_planar_ref = np.squeeze(p0), np.squeeze(p1)

	def plot_correspondences(self):
		fig = plt.figure()
		plt.subplot(2, 1, 1)
		plt.imshow(self.last_frame)
		plt.scatter(self.px_planar_ref[:, 0], self.px_planar_ref[:, 1], c='red')
		plt.subplot(2, 1, 2)
		plt.imshow(self.new_frame)
		plt.scatter(self.px_planar_cur[:, 0], self.px_planar_cur[:, 1], c='red')
		plt.show()
	def plot_error(self):
		fig = plt.figure()
		plt.plot(self.error)
		plt.grid()
		plt.xlabel('frame')
		plt.ylabel('scaling error')
		plt.title('error y_true - y_est')
		plt.show()