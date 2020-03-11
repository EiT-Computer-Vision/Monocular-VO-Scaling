import numpy as np
import cv2

from visual_odometry import PinholeCamera, VisualOdometry


cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
vo = VisualOdometry(cam, './dataset/poses/03.txt')

traj = np.zeros((600,600,3), dtype=np.uint8)

x, y, z = 0., 0., 0.
true_x1, true_y1, true_z1 = 0., 0., 0.
vx, vy, vz = 0., 0., 0.
true_vx, true_vy, true_vz = 0., 0., 0.

for img_id in range(5000): # Terminates on end of image sequence
	img = cv2.imread('./dataset/sequences/01/image_0/'+str(img_id).zfill(6)+'.png', 0)
	vo.update(img, img_id)
	cur_t = vo.cur_t
	Ts = 0.1
	if(img_id > 2):
		vx, vy, vz = (cur_t[0] - x) / Ts, (cur_t[1] - y) / Ts, (cur_t[2] - z) / Ts
		x, y, z = cur_t[0], cur_t[1], cur_t[2]
	draw_x, draw_y = int(x)+50, int(z)+90

	print('true velocity in x is: ', (vo.trueX - true_x1)/Ts)
	print('estimated velocity in x is: ', vx)
	print('true velocity in y is: ', (vo.trueY - true_y1) / Ts)
	print('estimated velocity in y is: ', vy)
	print('true velocity in z is: ',(vo.trueZ - true_z1) / Ts)
	print('estimated velocity in z is: ', vz)
	print('')

	true_x1, true_y1, true_z1 = vo.trueX, vo.trueY, vo.trueZ
	true_x, true_y = int(true_x1)+50, int(true_z1)+90 # translate to fit to frame

	cv2.circle(traj, (draw_x,draw_y), 1, (img_id*255/4540,255-img_id*255/4540,0), 1)
	cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 2)
	cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
	#text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x, y, z)
	text = "current velocity: x=%2fm/s y=%2fm/s z=%2fm/s"%(vx, vy, vz)
	cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

	cv2.imshow('Road facing camera', img)
	cv2.namedWindow('Trajectory', cv2.WINDOW_NORMAL)
	cv2.imshow('Trajectory', traj)
	cv2.resizeWindow('Trajectory', 1000,1000)
	cv2.waitKey(1)

cv2.imwrite('map.png', traj)