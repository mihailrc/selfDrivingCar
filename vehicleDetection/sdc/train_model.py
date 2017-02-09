import glob
import time
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from vehicleDetection.sdc.feature_extractor import *
import pickle

# images are 64x64 pixels
cars = glob.glob('./../vehicles/*/*png')
# cars = glob.glob('./../vehicles/GTI_Far/*png')
notcars = glob.glob('./../non-vehicles/*/*png')
print("There are {0} car images and {1} non-car images".format(len(cars),len(notcars)))

print("Extracting feature vectors from car images...")
car_features = extract_features_from_files(cars, color_space='YCrCb')
print("Extracting feature vectors from notcar images...")
not_car_features = extract_features_from_files(notcars, color_space='YCrCb')

# Stack the feature vectors
X = np.vstack((car_features, not_car_features)).astype(np.float64)
# Scale the features
X_scaler = StandardScaler().fit(X)
X_scaled = X_scaler.transform(X)

# Create the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

# Split into test and training sets
random_int = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=random_int)

# Wrap LinearSVC with a CalibtratedClassifier to get probabilities
classifier = CalibratedClassifierCV(LinearSVC())
print("Training the model ... ")
start=time.time()
classifier.fit(X_train, y_train)
end = time.time()
print('Trained the model in {0}'.format((end-start)))
# Check the score of the SVC
print('Train Accuracy = {0}'.format(classifier.score(X_train, y_train)))
print('Test Accuracy = {0}'.format(classifier.score(X_test, y_test)))

#Save both the classifier and the scaler
print("Saving models...")
pickle.dump(classifier, open('./../models/classifier2.pkl', "wb"))
pickle.dump(X_scaler, open('./../models/scaler2.pkl', "wb"))

print("Yupee!")
