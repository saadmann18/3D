import tensorflow.keras
import cv2
import numpy as np
import time


np.set_printoptions(suppress=True)
# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

with open('labels.txt', 'r') as f:
	class_names = f.read().split('\n')
print(class_names)
# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

size = (224, 224)

cap = cv2.VideoCapture(0)

while cap.isOpened:

	start = time.time()
	ret, image = cap.read()

	height, width, channels = image.shape
	scale_value = width/height
	image_resized = cv2.resize(image, size, fx=scale_value, fy=1, interpolation=cv2.INTER_NEAREST)
	image_array = np.asarray(image_resized)

	# Normalize the image
	normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
	# Load the image into the array
	data[0] = normalized_image_array

	# run the inference
	prediction = model.predict(data)

	index = np.argmax(prediction)
	class_name = class_names[index]
	confidence_score = prediction[0][index]

	end = time.time()
	totalTime = end - start

	fps = 1 / totalTime
	#print("FPS: ", fps)
	cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

	cv2.putText(image, class_name, (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
	cv2.putText(image, str(float("{:.2f}".format(confidence_score*100))) + "%", (75,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

	cv2.imshow('Classification Resized', image_resized)
	cv2.imshow('Classification Original', image)


	if cv2.waitKey(5) & 0xFF == 27:
	   break


cv2.destroyAllWindows()
cap.release()

