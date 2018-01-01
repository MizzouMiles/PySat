from __future__ import division
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
#import mpl_toolkits.mplot3d.axes3d as p3
from PIL import Image
import requests 			
import numpy as np
#from StringIO import StringIO # for python 2.x
import io # For python 3.x

def odeFunc(t, R):

	r2 = (R[[0]]**2 + R[[1]]**2 + R[[2]]**2)**(0.5)
        
    # Constants
	mu = 3.98716708E5 # Gravitational parameters [km^3/s^2]
	Re = 6378.137     # Radius of the earth
	J = 0.0010826267
	gamma = (5*R[2]**2)/(r2**2) # Partial coefficients used to calculate ap
	lambd = (-3*J*mu*Re**2)/(2*r2**5)

    # Pertubation due to non-spherical earth
	ap = np.reshape(np.array([[lambd*R[0]*(1-gamma)],[lambd*R[1]*(1-gamma)], [lambd*R[2]*(3-gamma)]]), (3,1))

	r_ddot = (-mu*R[0:3])/(r2**3) + ap # Acceleration

	ddy = [np.zeros((6,1))]
	
	ddy = np.reshape(np.array([[R[3]], [R[4]], [R[5]], [r_ddot[0]], [r_ddot[1]], [r_ddot[2]]]), (6,1)) # Returned values for velocity and acceleration [[dr/dt]; [d^2r/dt^2]]

	return ddy

def rungeKutta(func, tspan, steps, y0, order):

	[m, n] = np.shape(y0)
	dy = np.zeros((m, (steps + 1)))

	t  = np.zeros((1, (steps + 1)))

	dy[:,[0]] = y  = y0
	t[0]  = ti = tspan[0] 
	h = (tspan[1] - tspan[0]) / float(steps)
	
	if order not in [1, 2, 4]:

		print('Error! Order integer must be == 1, 2 or 4!')

		return

	# FIRST ORDER RUNGE-KUTTA
	for i in range(1, steps + 1): # ITERATE 
		
		if order == 1:

			k1 = odeFunc(ti, y)

			t[0,i]  = ti = tspan[0] + i * h
			dy[:,[i]] = y  = y + k1 * h
			
	## SECOND ORDER RUNGE-KUTTA
		if order == 2:

			k1 = odeFunc(ti, y) * h
			k2 = odeFunc(ti, y + k1 / 2) * h

			t[0,i]  = ti = tspan[0] + i * h
			dy[:,[i]] = y  = y + k2

	# FOURTH ORDER RUNGE-KUTTA
		if order == 4:

			k1 = odeFunc(ti, y) * h
			k2 = odeFunc(ti + h / 2, y + k1 / 2) * h
			k3 = odeFunc(ti + h / 2, y + k2 / 2) * h
			k4 = odeFunc(ti + h, y + k3) * h

			t[0,i]  = ti = tspan[0] + i * h
			dy[:,[i]] = y  = y + (k1 + 2 * (k2 + k3) + k4) / 6

	return(dy, t)

def drawEarth(Radius):

	# Create a sphere with earths surface texture

	# Load texture
	response = requests.get(
		'http://www.johnstonsarchive.net/spaceart/cmaps/earthmap.jpg')

	#img = Image.open(StringIO(response.content)) # For python 2.x
	img = Image.open(io.BytesIO(response.content)) # For python 3.x

	# Rescale RGB values
	img = np.array(img.resize([int(d/4) for d in img.size]))/256.

	# Image coordinates
	lons = np.linspace(-180, 180, img.shape[1]) * np.pi/180 
	lats = np.linspace(-90, 90, img.shape[0])[::-1] * np.pi/180 

	x = Radius*np.outer(np.cos(lons), np.cos(lats)).T
	y = Radius*np.outer(np.sin(lons), np.cos(lats)).T
	z = Radius*np.outer(np.ones(np.size(lons)), np.sin(lats)).T
	
	# Alternatively, create a simple sphere object (faster)
	# pi = np.pi
	# phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
	# x = Radius*np.sin(phi)*np.cos(theta)
	# y = Radius*np.sin(phi)*np.sin(theta)
	# z = Radius*np.cos(phi)

	return x, y, z, img


class structtype():
    
    pass



### USAGE AND ANIMATION EXAMPLE

tspan = [0, 24*3600] # Time span (over 24 hours)

y0 = np.array([[-2384.46],
	           [5729.01],
	           [3050.46],
	           [-7.36138],
	           [-2.98997],
	           [1.64354]]) # Initial positions and velocities [km], [km/s]

t_steps = 500 # Step size

yout = structtype()
tout = structtype()

yout.euler, tout.euler = rungeKutta(odeFunc, tspan, t_steps, y0, 1)
yout.rk2,  tout.rk2 = rungeKutta(odeFunc, tspan, t_steps, y0, 2)
yout.rk4,  tout.rk4 = rungeKutta(odeFunc, tspan, t_steps, y0, 4)

fig = plt.figure(1)
ax = Axes3D(fig)

ax.plot(yout.euler[0,:], yout.euler[1,:], yout.euler[2,:], '--k')
ax.plot(yout.rk2[0,:], yout.rk2[1,:], yout.rk2[2,:], '.-r')
ax.plot(yout.rk4[0,:], yout.rk4[1,:], yout.rk4[2,:], '-g')

x, y, z, bm = drawEarth(6378.137)

ax.plot_surface(
	    x, y, z,  rstride=4, cstride=4, alpha=0.5, facecolors=bm)
plt.legend(['Euler`s method', '2nd Order RK', '4th Order RK'])
ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')
plt.axis('square')
plt.show()


## ANIMATED ORBITAL TRAJECTORY 

def update_lines(num, dataLines, lines):

    for line, data in zip(lines, dataLines):
        
        line.set_data(data[0:2,:num])
        line.set_3d_properties(data[2,:num])

    return lines



fig2 = plt.figure(2)
ax2 = Axes3D(fig2)

data = [yout.euler[0:3, :] for index in range(t_steps)]

lines = [ax2.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], '-sr')[0] for dat in data]

ax2.set_xlim3d([-10000, 10000])
ax2.set_xlabel('X [km]')

ax2.set_ylim3d([-10000, 10000])
ax2.set_ylabel('Y [km]')

ax2.set_zlim3d([-10000, 10000])
ax2.set_zlabel('Z [km]')

ax2.set_title('3D Oribital Trajectory Animation')

ax2.plot_surface(
	    x, y, z,  rstride=4, cstride=4, alpha=0.5, facecolors=bm)

line_ani = animation.FuncAnimation(fig2, update_lines, int(t_steps/2), fargs=(data, lines),
                                   interval=t_steps, blit=False)

#line_ani.save('orbit_rk4.gif', dpi=80, writer='imagemagick') # Uncomment to generate/save animation

plt.show()