
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog
import cv2
from PyQt5 import uic 
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QDoubleValidator
import sys
import csv
import numpy as np
import math

lego_unit_size = 8 #mm

class UI(QMainWindow):
	colour_list = []
	def __init__(self):
		super(UI, self).__init__()

		self.load_csv()
		
		# Load the ui file
		uic.loadUi("main.ui", self)
		self.previous_desired_width = 0

		# Click The Dropdown Box
		self.gui_file_browser_btn.clicked.connect(self.open_file_dialog)
		self.gui_min_lego_dim.valueChanged.connect(self.update_rendering)
		self.gui_desired_output_width_input.setValidator(QDoubleValidator(4,1000,2))
		self.gui_desired_output_width_input.textChanged.connect(self.update_rendering)

		self.input_image = cv2.imread("turtle.jpg")
		self.set_open_cv_image_to_qlabel(self.input_image, self.gui_input_image)
		self.update_rendering()
						
		# Show The App
		self.show()

	def load_csv(self):
		with open('colors.csv', newline='') as csvfile:
			dict_reader = csv.DictReader(csvfile)
			for line in dict_reader:
				self.colour_list.append(line)
		
		for c in self.colour_list:
			c['rgb_array'] = bytearray.fromhex(c['rgb'])

			single_pixel_image = np.zeros((1,1,3), np.uint8)
			single_pixel_image[0,0] = c['rgb_array']

			out_lab = cv2.cvtColor(single_pixel_image, cv2.COLOR_RGB2LAB)
			c['lab_array'] = out_lab[0,0]
			c['lab_array_norm'] = self.lab_normalization(c['lab_array'])

	def lab_normalization(self, arr1):
		out = []
		out.append(arr1[0]*100/256)
		out.append(arr1[1] - 128)
		out.append(arr1[2] - 128)
		return out
	
	def compute_lab_distance(self, arr1, arr2):
		arr3 = []
		arr3.append(arr1[0]-arr2[0])
		arr3.append(arr1[1]-arr2[1])
		arr3.append(arr1[2]-arr2[2])
		return math.sqrt(arr3[0] ** 2 + arr3[1] ** 2+ arr3[2] ** 2)

	def find_closest_colour(self, colour_opencv_lab):
		lab_colour = self.lab_normalization(colour_opencv_lab)

		selected_colour = self.colour_list[0]
		dist = self.compute_lab_distance(self.colour_list[0]['lab_array_norm'], lab_colour)

		for c in self.colour_list:
			dist_loop = self.compute_lab_distance(c['lab_array_norm'], lab_colour)
			if dist_loop < dist:
				dist = dist_loop
				selected_colour = c
		return selected_colour

	def update_rendering(self):
		try:
			print("Rendering")
			self.update_rendering_work()
		except Exception as e:
			print(e)
			print("failed to render")
	
	def convert_image_to_lego_colours(self, image):
		self.lego_pieces = {}
		lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
		height, width, depth = lab_image.shape

		for i in range(0, height):
			for j in range(0, width):
					colour = self.find_closest_colour(lab_image[i,j])
					lab_image[i,j] = colour['lab_array']
					if colour['id'] in self.lego_pieces:
						self.lego_pieces[colour['id']]['count'] = self.lego_pieces[colour['id']]['count'] + 1
					else:
						self.lego_pieces[colour['id']] = colour
						self.lego_pieces[colour['id']]['count'] = 0

		with open('lego_bom.csv', 'w', newline='') as csv_file: 
			writer = csv.DictWriter(csv_file, fieldnames=['id','name','rgb','count','is_trans', 'lab_array', 'lab_array_norm', 'rgb_array'])
			writer.writeheader()
			for key, value in self.lego_pieces.items():
				writer.writerow(value)	

		return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)


	def update_rendering_work(self):
		input_image_h_px, input_image_w_px, ch = self.input_image.shape
		aspect_ratio = input_image_w_px/input_image_h_px
		
		output_picture_width = float(self.gui_desired_output_width_input.text())*10

		if self.previous_desired_width == output_picture_width:
			return
		
		if output_picture_width < 40:
			return
		
		self.previous_desired_width = output_picture_width
		output_picture_height = output_picture_width / aspect_ratio



		local_lego_unit_size = lego_unit_size
		if int(self.gui_min_lego_dim.text()) == 4:
			local_lego_unit_size = lego_unit_size * 2
		
		number_of_lego_width = int(output_picture_width / local_lego_unit_size)
		number_of_lego_height = int(output_picture_height / local_lego_unit_size)

		self.gui_output_width.setText(str(number_of_lego_width*local_lego_unit_size/10.0))	
		self.gui_output_height.setText(str(number_of_lego_height*local_lego_unit_size/10.0))	

		total_lego_pieces = number_of_lego_width * number_of_lego_height
		self.gui_number_of_pieces.setText(str(total_lego_pieces) +" (" +str(number_of_lego_width) + "x" + str(number_of_lego_height)  +")" )

		dim =  (number_of_lego_width, number_of_lego_height)

		resized_image = cv2.resize(self.input_image, dim, interpolation=cv2.INTER_CUBIC)

		lego_image = self.convert_image_to_lego_colours(resized_image)


		self.gui_rendered_image.setPixmap(QPixmap.fromImage(QImage(lego_image.data, lego_image.shape[1], lego_image.shape[0],number_of_lego_width*3, QImage.Format_RGB888).rgbSwapped()).scaledToWidth(input_image_w_px))

	def open_file_dialog(self):
		fname = QFileDialog.getOpenFileName(self, "Open File", "", "Image Files (*.png, *.jpg);;PNG Files (*.png);;Jpg Files (*.jpg);;All Files (*)")
		self.gui_file_name.setText(fname[0])

		# Open The Image
		if fname:
			self.input_image = cv2.imread(fname[0])
			self.set_open_cv_image_to_qlabel(self.input_image, self.gui_input_image)
			self.update_rendering()
			
	def set_open_cv_image_to_qlabel(self, open_cv_image, qlabel):
		qlabel.setPixmap(QPixmap.fromImage(QImage(open_cv_image.data, open_cv_image.shape[1], open_cv_image.shape[0], QImage.Format_RGB888).rgbSwapped()))
		
# Initialize The App
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()