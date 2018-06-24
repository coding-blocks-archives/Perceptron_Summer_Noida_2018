import urllib
import cv2
import numpy as np
import time


class Recognizer:

	def __init__(self, url=""):
		self.url = url
		self.area_threshold = 400
		self.contour_size_threshold = 100

	def get_frame(self):
		img_response = urllib.urlopen(self.url)
		img_numpy = np.array(bytearray(img_response.read()), dtype=np.uint8)
		img = cv2.imdecode(img_numpy, -1)
		return img

	def pre_process(self, img):
		# Apply Gaussian Blur
		im = cv2.GaussianBlur(img, (5, 5), 10)
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

		# Apply image thresholding
		th = 255 - cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

		kernel = np.ones((3, 3)) / 9.0

		# Apply some morphological operations on the image
		processed = cv2.dilate(th, kernel, iterations=1)
		processed = cv2.erode(processed, kernel, iterations=1)
		processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
		processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

		# Find contours in the image and sort based on size
		im, cnts, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))

		for cnt in cnts:
			if cv2.contourArea(cnt) > self.area_threshold:
				x, y, w, h = cv2.boundingRect(cnt)
				if np.abs(w - h) < self.contour_size_threshold:
					cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		return img

	def run(self):
		while True:
			# Get frame
			fr = self.get_frame()

			# Get processed image
			pr_fr = self.pre_process(fr)

			# Display image
			cv2.imshow('Digits', pr_fr)

			if cv2.waitKey(1) == ord('q'):
				break




if __name__ == '__main__':
	pass
	# Replace the URL with your own IPwebcam shot.jpg IP:port
	url='http://100.120.254.114:8080/shot.jpg'

	rec = Recognizer(url=url)
	rec.run()