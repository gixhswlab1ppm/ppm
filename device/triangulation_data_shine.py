import ujson
import numpy as np
import math
from numpy import linalg as LA
mpu = ujson.load(open('1548719339.7204287.json','r'))
mpu = np.asarray(mpu)

datas = mpu[:,[7,8,9]]


class circle(object):
	def __init__(self,point,radius):
		self.center = point
		self.radius = radius

class point(object):
	def __init__(self,lat,lon):
		self.x = lat
		self.y = lon
		#self.dist = dist

def get_two_points_distance(p1,p2):
	return math.sqrt(pow(p1.x-p2.x,2)+pow(p1.y-p2.y,2))

def angle1(c1,c2):
	dis = 0.5* (c1.radius-c2.radius+get_two_points_distance(c1.center,c2.center))
	print(dis,c1.radius)
	angle = math.acos(dis/c1.radius)
	return angle

def angle2(c1,c2):
	dis = get_two_points_distance(c1.center,c2.center)
	d_circles = get_two_points_distance(c1.center,c2.center)
	angle = math.acos(dis/d_circles)
	return angle

def get_two_circles_intersecting_points(c1,c2):
	d = get_two_points_distance(c1.center,c2.center)
	if d>= (c1.radius+c2.radius) or d<= math.fabs(c1.radius-c2.radius):
		return None
	ang_1 = angle1(c1,c2)
	ang_2 = angle2(c1,c2)
	px1 = c1.center.x+c1.radius*math.cos(ang_1+ang_2)
	py1 = c1.center.y+c1.radius*math.sin(ang_1+ang_2)

	px2 = c1.center.x+c1.radius*math.cos(abs(ang_1-ang_2))
	py2 = c1.center.y+c1.radius*math.sin(abs(ang_1-ang_2))
	#print("points",point(px1,py1),point(px2,py2))
	return [point(px1,py1),point(px2,py2)]



def triangular(c1,c2,c3):
	circles = []
	p1 = get_two_circles_intersecting_points(c1,c2)
	p2 = get_two_circles_intersecting_points(c2,c3)
	if p1 is not None:
		circles.extend(p1)
	if p2 is not None:
		if p2[0].x != p1[0].x and p2[1].y != p2[1].y:
			circles.extend(p2)
	return circles


if __name__ == '__main__':

	location={
	0:(0,10),
	1:(10,10),
	2:(5,0)
	}

	p1 = point(0,10)
	p2 = point(10,10)
	p3 = point(5,0)
	new_data = []
	for i in range(300):
		mask = [ x >0.5 for x in datas[i]]
		if False not in mask:
			new_data.append(datas[i])
	new_data = np.asarray(new_data)
	print("new_data length",len(new_data))

	for i,data in enumerate(new_data):
		circles = []
		c1 = circle(p1,data[0])
		c2 = circle(p2,data[1])
		c3 = circle(p3,data[2])
		print("c1",c1.radius,(c1.center.x,c1.center.y))
		print("c2",c2.radius,(c2.center.x,c2.center.y))
		print("c3",c3.radius,(c3.center.x,c3.center.y))
		circles = triangular(c1,c2,c3)
	for c in circles:
		print("circle",c.x,c.y)
	#print("circles",circles)
		# for i in range(len(circles)):
		# 	print(circles)
		#print(np.asarray(circles).shape)
		# if len(circles) != 0 :
			# print()
			# print("circle1",circles[0][0].x,circles[0][0].y)
			# print()
			# print("circle2",circles[0][1].x,circles[0][1].y)

		
	
	












