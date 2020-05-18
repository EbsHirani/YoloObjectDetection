import cv2
import numpy as np
IMAGE = "image.jpg"
confidences = []
boxes = []
class_ids = []
labels = []
yolo = cv2.dnn.readNet("/collegeWork/YoloObjectDetection/yolov3.weights", "yolov3.cfg")

with open("coco.names.txt", "r") as f:
	classes = [line.strip() for line in f.readlines()]

layers = yolo.getLayerNames()
outputlayers = [layers[i[0]-1] for i in yolo.getUnconnectedOutLayers()]



img = cv2.imread(IMAGE)

# img = cv2.resize(img, None, fx = 0.5, fy = 0.5)

height, width, _ = img.shape
blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop = False)

yolo.setInput(blob)

outputs = yolo.forward(outputlayers)

for out in outputs:
	for detection in out:
		scores = detection[5:]
		class_id = np.argmax(scores)
		# confidence = scores[class_id]
		confidence = detection[4] #output index 4 gives how confident the algorithm is with the bounding box		
		if confidence > 0.7:      #scores gives a measure of how confident the algorithm is with the class of the object.
			center_x = int(detection[0]*width)
			center_y = int(detection[1]*height)
			w = int(detection[2]*width)
			h = int(detection[3]*width)
			x = int(center_x - w/2)
			y = int(center_y - h/2)
			# cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
			boxes.append([x, y, w, h])
			confidences.append(float(confidence))
			class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.5)

for i, box in enumerate(boxes): #Display cropped images 
	if i in indexes:
		x, y, w, h = box
		cv2.imshow(str(classes[class_ids[i]]), img[y:y+h, x:x+w])

for i, box in enumerate(boxes): #Display boxes overlayed on the original image(not done in the same loop as the 
	if i in indexes:			#rectangles of other detectiond are also displayed in the cropped images)
		x, y, w, h = box
		label = str(classes[class_ids[i]])
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		cv2.putText(img, label, (x+w//2, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)

cv2.imshow("YOLO", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
