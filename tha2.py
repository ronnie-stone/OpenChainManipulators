import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os 
import glob
from PIL import Image


class Kinematics():

	def __init__(self):
		pass

	def make_video(self, path, savename):

		CURR_DIR = os.path.dirname(os.path.realpath(__file__))

		frames = []

		for filename in sorted(glob.glob(CURR_DIR + path), key=os.path.getmtime):
			new_frame = Image.open(filename)
			frames.append(new_frame)

		frames[0].save(savename + ".gif", format="GIF", append_images=frames[1:], save_all=True, loop=0)

	def vector_to_skewsymmetric_matrix(self, w):
		w_hat = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
		return w_hat

	def axis_angle_to_rotation_matrix(self, w_hat, w, theta):
		wnorm = np.linalg.norm(w)
		R = np.eye(3) + (w_hat/wnorm)*np.sin(wnorm*theta) + \
		(np.matmul(w_hat, w_hat)/wnorm**2)*(1 - np.cos(wnorm*theta))
		return R

	def plot_screw_axes(self, screw_axes, joint_loc, n_joints, ax):
		plt.sca(ax)
		for i in range(n_joints):
			s = screw_axes[:,i]
			d = joint_loc[i]
			ax.quiver(d[0],d[1],d[2],s[0],s[1],s[2], length=1, color="red")
		return

	def plot_frames(self, M, ax):
		plt.sca(ax)
		arrow_length = 0.5
		x_hat = np.array([1,0,0], dtype=float)
		y_hat = np.array([0,1,0], dtype=float)
		z_hat = np.array([0,0,1], dtype=float)
		ax.quiver(0,0,0,*x_hat, length=arrow_length, color="blue")
		ax.quiver(0,0,0,*y_hat, length=arrow_length, color="blue")
		ax.quiver(0,0,0,*z_hat, length=arrow_length, color="blue")
		R = M[0:3, 0:3]
		d = M[0:3, 3]
		x_body = np.matmul(R, x_hat)
		y_body = np.matmul(R, y_hat)
		z_body = np.matmul(R, z_hat)
		ax.quiver(*d, *x_body, length=arrow_length, color="green")
		ax.quiver(*d, *y_body, length=arrow_length, color="green")
		ax.quiver(*d, *z_body, length=arrow_length, color="green")
		return

	def plot_links(self, joint_loc, n_joints, M, ax):
		plt.sca(ax)
		for i in range(n_joints-1):
			ax.plot([joint_loc[i][0], joint_loc[i+1][0]], 
			[joint_loc[i][1], joint_loc[i+1][1]],
			[joint_loc[i][2], joint_loc[i+1][2]], "-kx")
		ax.plot([joint_loc[-1][0], M[0,3]],
			[joint_loc[-1][1], M[1,3]],
			[joint_loc[-1][2], M[2,3]], "-kx")
		return

	def plot_legend(self, ax):
		plt.sca(ax)
		ax.plot(0,0,color="r", label="Screw Axes")
		ax.plot(0,0,color="b", label="Spatial Frame")
		ax.plot(0,0,color="g", label="Body Frame")
		ax.plot(0,0,color="k", label="Manipulator")
		ax.legend(fontsize=18)
		return

	def rotation_matrix_to_axis_angle(self, R):

		# Check if rotation matrix is identity:

		if np.allclose(R, np.eye(3)):
			return np.zeros([3,3]), np.zeros(3), 0

		# Check edge case where trace is negative one:

		elif np.trace(R) == -1:
			theta = np.pi
			w = 1/(np.sqrt(2*(1+R[0][0])))*np.array([1+R[0][0], R[1][0], R[2][0]], dtype=float)
			w_hat = self.vector_to_skewsymmetric_matrix(w)
			return w_hat, w, theta

		# General case:

		else:
			theta = np.arccos(0.5*(np.trace(R)-1))
			w_hat = 1/(2*np.sin(theta))*(R - np.transpose(R))
			w = np.array([w_hat[2][1], w_hat[0][2], w_hat[1][0]], dtype=float)
			return w_hat, w, theta
	
	def find_transformations(self, screw_axes, joint_angles):
		n_joints = len(joint_angles)
		T_list = []
		for i in range(n_joints):
			T = np.zeros([4,4])
			w = screw_axes[:,i][0:3]
			v = screw_axes[:,i][3:6]
			theta = joint_angles[i]
			if np.allclose(w, np.zeros(3)):
				T[0:3, 0:3] = np.eye(3)
				T[0:3,3] = theta*v
			else:
				w_hat = self.vector_to_skewsymmetric_matrix(w)
				R = self.axis_angle_to_rotation_matrix(w_hat, w, theta)
				Ginv = np.eye(3)*theta + (1-np.cos(theta))*w_hat + \
				(theta-np.sin(theta))*np.matmul(w_hat, w_hat)
				tv = np.matmul(Ginv, v)
				T[0:3, 0:3] = R
				T[0:3,3] = tv
			T[-1,-1] = 1
			T_list.append(T)
		return T_list

	def find_home_configurations(self, joint_loc, ef_loc):
		M_list = []
		for i in range(1, len(joint_loc)):
			M = np.array([[1,0,0,joint_loc[i][0]], [0,1,0,joint_loc[i][1]], [0,0,1,joint_loc[i][2]], [0,0,0,1]], dtype=float)
			M_list.append(M)
		M_end = np.array([[1,0,0,ef_loc[0]], [0,1,0,ef_loc[1]], [0,0,1,ef_loc[2]], [0,0,0,1]], dtype=float)
		M_list.append(M_end)
		return M_list

	def update_joint_loc(self, joint_loc, new_joint_T):
		new_joint_loc = np.zeros_like(joint_loc)
		for i in range(len(joint_loc)-1):
			new_joint_loc[i+1] = new_joint_T[i][0:3,3]
		return new_joint_loc

	def update_screw_axes(self, screw_axes, new_joint_T):
		new_screw_axes = np.zeros_like(screw_axes)
		new_screw_axes[:,0][0:3] = screw_axes[:,0][0:3]
		for i in range(1, np.shape(screw_axes)[1]):
			new_screw_axes[0:3,i] = np.matmul(new_joint_T[i][0:3,0:3], np.transpose(screw_axes[:,i][0:3]))
		return new_screw_axes

	
	def adjoint_representation(self, T):
		R = T[0:3, 0:3]
		p = T[0:3, 3]
		p_hat = self.vector_to_skewsymmetric_matrix(p)
		AdjT = np.zeros([6,6], dtype=float)
		AdjT[0:3, 0:3] = R
		AdjT[3:6, 3:6] = R
		AdjT[3:6, 0:3] = np.matmul(p_hat, R)
		return AdjT
	
	def J_space(self, screw_axes, joint_angles):
		n_joints = len(joint_angles)
		Js = np.zeros([6,n_joints], dtype=float)
		Js[:,0] = screw_axes[:,0]
		T_list = self.find_transformations(screw_axes, joint_angles)
		for i in range(n_joints-1):
			T_start = T_list[0]
			Si = screw_axes[:,i+1]
			j = 1
			while j < i+1:
				T_start = np.matmul(T_start, T_list[j])
				j += 1
			AdjT = self.adjoint_representation(T_start)
			Js[:,i+1] = np.matmul(AdjT, Si)
		return Js
	
	def J_body(self, screw_axes, joint_angles):
		n_joints = len(joint_angles)
		Jb = np.zeros([6,n_joints], dtype=float)
		Jb[:,-1] = screw_axes[:,-1]
		T_list = self.find_transformations(screw_axes, -joint_angles)
		for i in range(n_joints-1, 0, -1):
			T_start = T_list[-1]
			Si = screw_axes[:,i-1]
			j = n_joints-3
			while j >= i-1:
				T_start = np.matmul(T_start, T_list[j])
				j -= 1
			AdjT = self.adjoint_representation(T_start)
			Jb[:,i-1] = np.matmul(AdjT, Si)
		return Jb 
	
	def ellipsoid_plot_angular(self, J, display=False):
		Jw = J[0:3, :]
		A = np.matmul(Jw, Jw.transpose())
		eigval, _ = np.linalg.eig(A)
		u = np.linspace(0, 2*np.pi, 100)
		v = np.linspace(0, np.pi)
		x = eigval[0] * np.outer(np.cos(u), np.sin(v))
		y = eigval[1] * np.outer (np.sin(u), np.sin(v))
		z = eigval[2] * np.outer(np.ones_like(u), np.cos(v))

		if display:
			fig = plt.figure(figsize=(15,10))
			ax = fig.add_subplot(111, projection = "3d")
			axes_size = 5
			ax.set_xlim(-axes_size,axes_size)
			ax.set_ylim(-axes_size,axes_size)
			ax.set_zlim(-axes_size,axes_size)
			ax.quiver(0,0,0,1,0,0, length=2, color="red")
			ax.quiver(0,0,0,0,1,0, length=2, color="red")
			ax.quiver(0,0,0,0,0,1, length=2, color="red")
			ax.plot_surface(x, y, z, rstride=4, cstride=4, color="b", alpha=0.3)
			ax.set_axis_off()
			plt.show()
		return
		
	def ellipsoid_plot_linear(self, J, display=False):
		Jv = J[3:6, :]
		A = np.matmul(Jv, Jv.transpose())
		eigval, _= np.linalg.eig(A)
		u = np.linspace(0, 2*np.pi, 100)
		v = np.linspace(0, np.pi)
		x = eigval[0] * np.outer(np.cos(u), np.sin(v))
		y = eigval[1] * np.outer (np.sin(u), np.sin(v))
		z = eigval[2] * np.outer(np.ones_like(u), np.cos(v))

		if display:
			fig = plt.figure(figsize=(15,10))
			ax = fig.add_subplot(111, projection = "3d")
			axes_size = 5
			ax.set_xlim(-axes_size,axes_size)
			ax.set_ylim(-axes_size,axes_size)
			ax.set_zlim(-axes_size,axes_size)
			ax.quiver(0,0,0,1,0,0, length=2, color="red")
			ax.quiver(0,0,0,0,1,0, length=2, color="red")
			ax.quiver(0,0,0,0,0,1, length=2, color="red")
			ax.plot_surface(x, y, z, rstride=4, cstride=4, color="b", alpha=0.3)
			ax.set_axis_off()
			plt.show()
		return

	def J_isotropy(self, J):
		Jw = J[0:3, :]
		Jv = J[3:6, :]
		Aw = np.matmul(Jw, Jw.transpose())
		Av = np.matmul(Jv, Jv.transpose())
		eigvalW, _ = np.linalg.eig(Aw)
		eigvalV, _ = np.linalg.eig(Av)
		isotropyW = np.sqrt(np.max(eigvalW)/np.min(eigvalW))
		isotropyV = np.sqrt(np.max(eigvalV)/np.min(eigvalV))
		return isotropyW, isotropyV
	
	def J_condition(self, J):
		isotropyW, isotropyV = self.J_isotropy(J)
		return isotropyW**2, isotropyV**2
	
	def J_ellipsoid_volume(self, J):
		Jw = J[0:3, :]
		Jv = J[3:6, :]
		Aw = np.matmul(Jw, Jw.transpose())
		Av = np.matmul(Jv, Jv.transpose())
		volumeW = np.linalg.det(Aw)
		volumeV = np.linalg.det(Av)
		return volumeW, volumeV

	def pose_error(self, current_pose, desired_pose):
		R_error = current_pose[0:3,0:3] - desired_pose[0:3,0:3]
		p_error = current_pose[0:3] - desired_pose[0:3]
	
	def J_inverse_kinematics(self, ef_loc, joint_loc, joint_loc_b, joint_axes, 
		joint_type, T_sd, angle_guess, max_iter=20, epsW=0.001, epsV=0.0001, display=False):

		angles_matrix = []
		angles_matrix.append(angle_guess)
		i = 0 

		while i < max_iter:

			current_angles = angles_matrix[i]
			print(angle_guess)

			if display:
				S, T_sb, M = self.FK_spatial(ef_loc, joint_loc, joint_axes,
				joint_type, current_angles, display=True, frame_number=i)

			B, T_sb, M = self.FK_spatial(ef_loc, joint_loc_b, joint_axes, joint_type, current_angles, frame="body", display=False)
			T_bd = np.matmul(np.linalg.inv(T_sb), T_sd)
			R_bd = T_bd[0:3,0:3]
			p_bd = T_bd[0:3,3]
			w_hat, w, theta = self.rotation_matrix_to_axis_angle(R_bd)
			if theta == 0:
				theta = np.linalg.norm(p_bd)
				v = p_bd/theta
			else:
				Ginv = (1/theta)*np.eye(3) - 0.5*w_hat + \
				(1/theta - 0.5/(np.tan(theta/2)))*np.matmul(w_hat, w_hat)
				v = np.matmul(Ginv,p_bd) 
			V_error = np.array([w[0], w[1], w[2], v[0], v[1], v[2]])*theta
			w_error = V_error[0:3]
			v_error = V_error[3:6]

			if np.linalg.norm(w_error) < epsW and np.linalg.norm(v_error < epsV):
				print("CONVERGED")
				break

			Jb = self.J_body(B, current_angles)
			Jb_transpose = np.transpose(Jb)
			m,n = Jb.shape
			print(m,n)

			if n < m:
				J_pseudo = np.matmul(np.linalg.inv(np.matmul(Jb_transpose, Jb)), Jb_transpose)
			elif n > m:
				J_pseudo = np.matmul(Jb_transpose, np.linalg.inv(np.matmul(Jb, Jb_transpose)))
			else:
				J_pseudo = np.linalg.inv(Jb)

			current_angles = current_angles + np.matmul(J_pseudo, V_error)

			angles_matrix.append(current_angles)
			print("Iteration no. " + str(i))
			print("Current_Angles:" + str(current_angles*180/np.pi))
			i += 1

		S, T_sb, M = self.FK_spatial(ef_loc, joint_loc, joint_axes,
				joint_type, current_angles, display=True, frame_number=i+1)
		self.make_video("/ik_images/*.png", "ik_GIF")

	def FK_spatial(self, ef_loc, joint_loc, joint_axes, joint_type, joint_angles,
	display=False, frame="space", frame_number=0):

		n_joints = len(joint_type)
		screw_axes = np.zeros([6,n_joints], dtype=float)
		
		# First we want to calculate the screw axes: 

		for i in range(n_joints):
			if joint_type[i] == 0:
				v_i = np.cross(-joint_axes[i], joint_loc[i])
				screw_axes[0:3, i] = joint_axes[i]
				screw_axes[3:6, i] = v_i
			else:
				screw_axes[3:6,i] = joint_axes[i] 

		# Second we calculate the new joint location of each joint and end-effector:

		M_list = self.find_home_configurations(joint_loc, ef_loc)
		T_list = self.find_transformations(screw_axes, joint_angles)
		new_joint_T = []
		k = 0
		for j in range(n_joints-1, -1, -1):
			T_end = M_list[j]
			for i in range(n_joints-1-k, -1, -1):
				T_end = np.matmul(T_list[i], T_end)
			k += 1
			new_joint_T.append(T_end)
		joint_loc = self.update_joint_loc(joint_loc, new_joint_T[::-1])
		new_screw_axes = self.update_screw_axes(screw_axes, new_joint_T[::-1])

		#print("Screw Axes in FK:")

		#print(screw_axes)

		#print("Transformation List in FK:")
		#print(T_list)

		if frame=="body":
			T_end = M_list[-1]
			for i in range(n_joints):
				T_end = np.matmul(T_end, T_list[i])
			new_joint_T[0] = T_end

		# Third we want to check if plotting was requested:

		if display:
			fig = plt.figure(figsize=(15,10))
			ax = fig.add_subplot(111, projection = "3d")
			axes_size = 10
			ax.set_xlim(-axes_size,axes_size)
			ax.set_ylim(-axes_size,axes_size)
			ax.set_zlim(-axes_size,axes_size)
			ax.view_init(elev=30, azim=45)
			self.plot_screw_axes(new_screw_axes, joint_loc, n_joints, ax)
			self.plot_frames(new_joint_T[0], ax)
			self.plot_links(joint_loc, n_joints, new_joint_T[0], ax)
			self.plot_legend(ax)
			fig.savefig("ik_images/iteration" + str(frame_number) + ".png")
			plt.close()

		return screw_axes, new_joint_T[0], M_list[-1]
		
if __name__ == "__main__": 

	robot_k = Kinematics()

	# 2R Robot:

	end_effector = np.array([2,0,0], dtype=float)
	joint_loc = np.array([[0,0,0], [1,0,0]], dtype=float)
	joint_loc_b = np.array([[-2,0,0], [-1,0,0]], dtype=float)
	joint_axes = np.array([[0,0,1], [0,0,1]], dtype=float)
	joint_type = np.array([0,0])
	joint_angles = np.array([np.pi/6, np.pi/2], dtype=float)

	S, T_se, M = robot_k.FK_spatial(end_effector, joint_loc, joint_axes, joint_type, joint_angles, display=False)
	B, T_be, M = robot_k.FK_spatial(end_effector, joint_loc_b, joint_axes, joint_type, joint_angles, display=False, frame="body")

	T_sd = np.array([[-0.5, -0.866, 0, 0.366], [0.866, -0.5, 0, 1.366], [0,0,1,0],[0,0,0,1]], dtype=float)
	angle_guess = np.array([0, np.pi/6], dtype=float)

	# robot_k.J_inverse_kinematics(end_effector, joint_loc_b, joint_axes, joint_type, T_sd, angle_guess)


	# 3R Robot:

	end_effector = np.array([3,0,0], dtype=float)
	joint_loc = np.array([[0,0,0], [1,0,0], [2,0,0]], dtype=float)
	joint_axes = np.array([[0,0,1], [0,0,1], [0,0,1]], dtype=float)
	joint_type = np.array([0,0,0])
	joint_angles = np.ones(3)*np.pi/4
	M = np.array([[1, 0, 0, 3], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
	# S, M = robot_k.FK_spatial(end_effector, joint_loc, joint_axes, joint_type, joint_angles, display=True)

	# RRRP Robot:

	end_effector = np.array([2,0,0])
	joint_loc = np.array([[0,0,0], [1,0,0], [2,0,0], [2,0,0]], dtype=float)
	joint_axes = np.array([[0,0,1], [0,0,1], [0,0,1], [0,0,1]], dtype=float)
	joint_type = np.array([0,0,0,1])
	# joint_loc_body = np.array([[-2,0,0], [-1,0,0], [0,0,0], [0,0,0]], dtype=float)
	joint_angles = np.array([np.pi/3, np.pi/2, 0, 0.5], dtype=float)
	# M = np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
	# S, T_end = robot_k.FK_spatial(end_effector, joint_loc, joint_axes, joint_type, joint_angles, display=True)
	# Js = robot_k.J_space(S, joint_angles)
	#print(Js)
	#Tbs = np.linalg.inv(T_end)
	#AdJBS = robot_k.adjoint_representation(Tbs)
	#Jb = np.matmul(AdJBS, Js) 

	# Yaskawa:

	end_effector = np.array([0, 11.25, 3], dtype=float)
	joint_loc = np.array([[0,0,0], [0,0,3], [0,2.5,3], [0,4.3,3], [0, 6.1, 3],
	[0, 7.9, 3], [0, 9.7, 3], [0, 11.25, 3]], dtype=float)
	joint_loc_b = np.array([[0, -11.25, -3], [0,-11.25,0], [0,-8.75,0], [0,-6.95,0],
	[0,-5.15,0], [0,-3.35,0], [0,-1.55,0], [0,0,0]], dtype=float)
	joint_axes = np.array([[0,0,1], [0,1,0], [0,0,1], [0,1,0], [0,0,1], [0,1,0], [0,0,1], [0,1,0]])
	joint_type = np.array([0,0,0,0,0,0,0,0])
	joint_angles = np.array([-np.pi/6,np.pi/4,np.pi/4,0,0,0,0,0])
	angle_guess = np.array([np.pi/2,np.pi/3,np.pi/3,np.pi/8,0,0,0,0])
	B, T_sd, M = robot_k.FK_spatial(end_effector, joint_loc_b, joint_axes, joint_type, joint_angles, display=False, frame="body", frame_number=0)
	robot_k.J_inverse_kinematics(end_effector, joint_loc, joint_loc_b, joint_axes, joint_type, T_sd, angle_guess, display=True)
	#S, M = robot_k.FK_spatial(end_effector, joint_loc, joint_axes, joint_type, joint_angles, T_sd, display=True)
	#Js = robot_k.J_space(S, joint_angles)
	#robot_k.ellipsoid_plot_angular(Js, display=False)
	#robot_k.ellipsoid_plot_linear(Js, display=False)
	#isoW, isoV = robot_k.J_isotropy(Js)
	#condW, condV = robot_k.J_condition(Js)
	#volumeW, volumeV = robot_k.J_ellipsoid_volume(Js)