from flask import Flask, render_template, redirect, request
import torch
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

# __name__ == __main__
app = Flask(__name__)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/yolov5ss.pt', force_reload=True)


@app.route('/')
def hello():
	return render_template("index.html")

@app.route('/', methods= ['POST'])
def marks():
	if request.method == 'POST':

		f = request.files['userfile']
		path = "./static/{}".format(f.filename)# ./static/images.jpg
		f.save(path)
		#caption = model(path)

		cap = cv2.VideoCapture(0)
		while cap.isOpened():
			ret,frame = cap.read()
			# Make detections 
			results = model(frame)
			x = results.pandas().xyxy[0]
			print(x)

			cv2.imshow('YOLO', np.squeeze(results.render()))
			if cv2.waitKey(10) & 0xFF == ord('q'):
				break
		cap.release()
		cv2.destroyAllWindows()
      
		
		result_dic = {
		'image' : path,
		'caption' : "hjasdj"
		}
	return render_template("index.html", your_result =result_dic)

if __name__ == '__main__':
	# app.debug = True
	# due to versions of keras we need to pass another paramter threaded = Flase to this run function
	app.run(debug = False, threaded = False)


