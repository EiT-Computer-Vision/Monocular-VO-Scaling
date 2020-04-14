import numpy as np
import cv2
from matplotlib import pyplot as plt
from visual_odometry import PinholeCamera, VisualOdometry

sequence = 3
height, width = np.shape(cv2.imread('../dataset/sequences/'+ str(sequence).zfill(2) + '/image_0/'+str(0).zfill(6)+'.png', 0))
cam = PinholeCamera(width, height, 718.8560, 718.8560, 607.1928, 185.2157)
vo = VisualOdometry(cam, '../dataset/poses/'+str(sequence).zfill(2)+'.txt')

window_width, window_height = 1000, 1000
traj = np.zeros((window_width, window_height,3), dtype=np.uint8)

x, y, z = 0., 0., 0.
x_1, y_1, z_1 = 0., 0., 0.
true_x1, true_y1, true_z1 = 0., 0., 0.

for img_id in range(len(vo.annotations)): # Terminates on end of image sequence
	img = cv2.imread('../dataset/sequences/'+ str(sequence).zfill(2) + '/image_0/'+str(img_id).zfill(6)+'.png', 0)
	vo.update(img, img_id)
	cur_t = vo.cur_t
	cur_t_u = vo.cur_t_unscaled
	if(img_id > 2):
		x, y, z = cur_t[0], cur_t[1], cur_t[2]
		x_1, y_1, z_1 = cur_t_u[0], cur_t_u[1], cur_t_u[2]
	offset_x, offset_z = 250, 150
	draw_x, draw_y = int(x) + offset_x, int(z) + offset_z
	draw_x1, draw_y1 = int(x_1) + offset_x, int(z_1) + offset_z
	true_x1, true_y1, true_z1 = vo.trueX, vo.trueY, vo.trueZ
	true_x, true_y = int(true_x1)+offset_x, int(true_z1) + offset_z # translate to fit to frame

	plt_scale = 4540
	cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 2)
	cv2.circle(traj, (draw_x,draw_y), 1, (img_id*255/plt_scale,255-img_id*255/plt_scale,0), 1 )
	#cv2.circle(traj, (draw_x1, draw_y1), 1, (255, 0, 0), 1)
	cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
	text = "Position: x=%2fm y=%2fm z=%2fm"%(x, y, z)
	cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

	cv2.imshow('Road facing camera', img)
	cv2.imshow('Trajectory', traj)
	cv2.waitKey(10)
vo.plot_error()
cv2.imwrite('map_sequence' + str(sequence) + '.png', traj)