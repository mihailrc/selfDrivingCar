import numpy as np
import cv2
import glob
import time
# from vehicleDetection.sdc.extract_features import Features
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from vehicleDetection.sdc.feature_extractor import *
import pickle

# All images are 64x64 pixels
cars = glob.glob('./../vehicles/*/*png')
# cars = glob.glob('./../vehicles/GTI_Far/*png')
notcars = glob.glob('./../non-vehicles/*/*png')

print("Found {} car images".format(len(cars)))
print("Found {} non-car images".format(len(notcars)))

print("Extracting feature vectors from car images...")
# car_f= Features()
# car_features = car_f.extract_image_file_features(cars, cspace='YCrCb')
car_features = extract_features_from_files(cars, color_space='YCrCb')
# print('Car features shape ', car_features.shape)
print("Extracting feature vectors from notcar images...")
# notcar_f= Features()
# notcar_features = notcar_f.extract_image_file_features(notcars, cspace='YCrCb')
notcar_features = extract_features_from_files(notcars, color_space='YCrCb')


# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
X_scaled = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=rand_state)

# Use a linear SVC
svc = LinearSVC()
# Add a calibrated classifier to get probabilities
clf = CalibratedClassifierCV(svc)
# Check the training time for the SVC
t=time.time()
print("Training Linear SVC Classifier...")
clf.fit(X_train, y_train)
t2 = time.time()
print(t2-t, 'Seconds to train SVC...')
# Check the score of the SVC
print('Train Accuracy of SVC = ', clf.score(X_train, y_train))
print('Test Accuracy of SVC = ', clf.score(X_test, y_test))
# Check the prediction time for a single sample
t=time.time()
prediction = clf.predict(X_test[0].reshape(1, -1))
prob = clf.predict_proba(X_test[0].reshape(1, -1))
t2 = time.time()
print(t2-t, 'Seconds to predict with SVC')
print("Prediction {}".format(prediction))
print("Prob {}".format(prob))

# Save model & scaler
print("Saving models...")
pickle.dump(clf, open('./../models/classifier2.pkl', "wb"))
pickle.dump(X_scaler, open('./../models/scaler2.pkl', "wb"))

# joblib.dump(svc, './../models/classifier2.pkl')
# joblib.dump(clf, './../models/calibrated2.pkl')
# joblib.dump(X_scaler, './../models/scaler2.pkl')
print("Done!")
