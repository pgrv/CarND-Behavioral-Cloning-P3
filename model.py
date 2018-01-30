import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
import sklearn

# Import the paths to the images and the steering angle from the csv file
def importImagesPath():
	lines = []

	# Read csv file
	with open('./data/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

	images = []
	measurements = []

	# Load center, left and right image and steering angle
	for line in lines:
		for i in range(3):
			source_path = line[i]
			filename = source_path.split('/')[-1]
			current_path = './data/IMG/' + filename
			images.append(current_path)
			correction = 0.2
			measurement = float(line[3])
			if(i==1):
				measurement = measurement + correction
			if(i==2):
				measurement = measurement - correction
			measurements.append(measurement)
	return images, measurements

# Use Nvidia model
def createModel():
	model = Sequential()
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((70,25),(0,0))))
	model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
	model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
	model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
	model.add(Convolution2D(64, 3, 3, activation="relu"))
	model.add(Convolution2D(64, 3, 3, activation="relu"))
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))

	return model

# Define the python generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                image = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)

                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


## Main program
# Read images path and steering angle
images, measurements = importImagesPath()

# Split samples
samples = list(zip(images, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=2)
validation_generator = generator(validation_samples, batch_size=2)
model = createModel()
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)

# Save the model
model.save('model.h5')
exit()