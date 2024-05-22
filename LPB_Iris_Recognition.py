import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from skimage import feature
import matplotlib.pyplot as plt
import glob
import time
import joblib
import logging
import sys

# Config logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("LBP_Iris_Recognition.log"),
                        logging.StreamHandler(sys.stdout)
                    ])

logging.info("Starting the iris recognition script")

class LBPIrisRecognition:
    def __init__(self, numPoints, radius, method="default"): 
        self.numPoints = numPoints
        self.radius = radius
        self.method = method

    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method=self.method) #LBP
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        return hist

def load_iris_dataset(inputPath, minSamples=10):
    imagePaths = glob.glob(os.path.join(inputPath, "*", "*.jpg"))
    labels = [p.split(os.path.sep)[-2] for p in imagePaths]
    (unique_labels, counts) = np.unique(labels, return_counts=True)
    valid_labels = unique_labels[counts >= minSamples]


    images = []
    labels_list = []
    
    logging.info(f"Found {len(imagePaths)} images in total.")
    logging.info(f"Found {len(unique_labels)} unique labels.")
    logging.info(f"Valid labels: {valid_labels}")


    for label in valid_labels:
        label_path = os.path.join(inputPath, label)
        label_images = glob.glob(label_path + "/*.jpg")
        
        logging.info(f"Processing label: {label}, found {len(label_images)} images")
        
        for imagePath in imagePaths:
            image = preprocess_img(imagePath)
            images.append(image)
            labels_list.append(label)

        # if counts[labels.index(name)] < minSamples:
        #     continue

    images = np.array(images)
    labels_list = np.array(labels_list)
    
    logging.info(f"Loaded {len(images)} images and {len(labels_list)} labels after filtering.")

    return (images, labels_list)

def save_dataset(images, labels, images_path='images.joblib', labels_path='labels.joblib'):
    joblib.dump(images, images_path)
    joblib.dump(labels, labels_path)
    logging.info("Dataset saved to disk.")

def load_saved_dataset(images_path='images.joblib', labels_path='labels.joblib'):
    images = joblib.load(images_path)
    labels = joblib.load(labels_path)
    logging.info("Dataset loaded from disk.")
    return (images, labels)

# Function for model extraction and training
def train_and_evaluate(model, X_train, y_train, X_test, y_test, method_name):
    logging.info(f"Training model with {method_name} LBP")
    start_time = time.time()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    end_time = time.time()
    logging.info(f"Training time for {method_name} LBP: {end_time - start_time} seconds")

    f1 = f1_score(y_test, y_pred, average="macro")
    logging.info(f"The F1-score of {method_name} LBP is: {f1}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (fn + tp)
    logging.info(f"False Positive Rate: {false_positive_rate}")
    logging.info(f"False Negative Rate: {false_negative_rate}")

    logging.info(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, f"iris_recognition_model_{method_name}.joblib")
    logging.info(f"Model saved as iris_recognition_model_{method_name}.joblib")

def preprocess_img(imagePath): #IRIS Estimation?
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    return image

def compare_images(lbp_descriptor, dataset_images, labels, label_to_compare, image_index_to_compare, my_image_path_to_compare, hist_metric=cv2.HISTCMP_CHISQR):
    if not (0 <= label_to_compare < len(np.unique(labels))):
        print("Enter a correct label from 0 to", len(np.unique(labels)) - 1)
        return

    if not (0 <= image_index_to_compare < 10):
        print("Enter a correct image index from 0 to 9")
        return

    # Find the dataset image corresponding to the provided label and image index
    matching_images = [img for img, lbl in zip(dataset_images, labels) if lbl == label_to_compare]

    if len(matching_images) <= image_index_to_compare:
        print("The specified image index is out of range for the given label")
        return

    dataset_image = matching_images[image_index_to_compare]

    # Describe the dataset image using LBP
    dataset_hist = lbp_descriptor.describe(dataset_image)

    # Assuming `image_path` is the path to the user-provided image
    user_image_hist = lbp_descriptor.describe(cv2.imread(my_image_path_to_compare, cv2.IMREAD_GRAYSCALE))

    # Compare histograms
    similarity = cv2.compareHist(np.array(dataset_hist, dtype=np.float32), np.array(user_image_hist, dtype=np.float32), hist_metric)
    return similarity


radius = 3
numPoints = 8 * radius #8px

dataset_path = 'CASIA Iris Syn'
images_path = 'images.joblib'
labels_path = 'labels.joblib'

# Initialize the LBP descriptor for different methods
uniformLBP = LBPIrisRecognition(numPoints, radius, method="uniform")
rorLBP = LBPIrisRecognition(numPoints, radius, method="ror")
varLBP = LBPIrisRecognition(numPoints, radius, method="var")
defaultLBP = LBPIrisRecognition(numPoints, radius, method="default")

# Check if the dataset is already saved or load it
if os.path.exists(images_path) and os.path.exists(labels_path):
    images, labels = load_saved_dataset(images_path, labels_path)
else:
    images, labels = load_iris_dataset(dataset_path)
    save_dataset(images, labels, images_path, labels_path)

print(len(labels))  # This should now print 10000 for 1000 labels with 10 images each
logging.info(f"Total number of labels: {len(labels)}")

# image_path = "E:\Minakova\Diploma\archive\000\S6000S09.jpg"
# label_to_compare = 0
# image_index = 8
# hist_metric = cv2.HISTCMP_CORREL

# similarity_uniform = compare_images(uniformLBP, images, labels, label_to_compare, image_index, hist_metric)
# print(f"Similarity (Uniform LBP, Chi-Square): {similarity_uniform}")


# # Encode the labels
# le = LabelEncoder()
# labels = le.fit_transform(labels)

# # Split the data into training and test set
# X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.20, stratify=labels, random_state=42)

# # Initialising the SVM model
# model = SVC(kernel="linear", C=100.0, random_state=42)

# # Extracting and training models
# #methods = [("uniform", uniformLBP), ("ror", rorLBP), ("var", varLBP), ("default", defaultLBP)]
# methods = [("uniform", uniformLBP)]

# try:
#     # for method_name, lbp in methods:
#     #     X_train_lbp = [lbp.describe(x) for x in X_train]
#     #     X_test_lbp = [lbp.describe(x) for x in X_test]
#     #     train_and_evaluate(model, X_train_lbp, y_train, X_test_lbp, y_test, method_name)
# except KeyboardInterrupt:
#     logging.info("Training interrupted. Saving current progress.")
#     # Saving the model before interruption
#     joblib.dump(model, "interrupted_model.joblib")
#     logging.info("Interrupted model saved as interrupted_model.joblib")

logging.info("Script completed")