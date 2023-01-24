import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
from scipy import spatial
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from detector import Detector
import cv2

# function to save plotted figures
def plot_figure(dictionary, model_name, save=False):
    rows = 5
    columns = 5
    count = 1
    fig = plt.figure(figsize=(columns, rows), tight_layout=False)
    for label in dictionary:
        for image in dictionary[label]:
            fig.add_subplot(rows, columns, count)
            # change BGR order to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.title(label)
            plt.axis('off')
            count+=1
    if save:
        plt.savefig('assets/'+ model_name +'.png')
    else:
        plt.show()

# normalize images
def prewhiten(image_array):
    axis = (0, 1, 2)
    size = image_array.size
    mean = np.mean(image_array, axis=axis, keepdims=True)
    std = np.std(image_array, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (image_array - mean) / std_adj
    return y

def create_dictionary(faces_features, face_boxes):
    # creating dictionary for easier plotting
    dictionary = dict()
    for i in range(len(faces_features)):
        if faces_features[i] not in dictionary:
            dictionary[faces_features[i]] = [face_boxes[i]]
        else:
            dictionary[faces_features[i]].append(face_boxes[i])
    # length of dictionary is number of unique faces
    print(len(dictionary), "unique faces found.")
    return dictionary

def predict(faces, model_name):
    model = tf.keras.models.load_model(model_name)
    face_boxes_resized = faces.copy()
    outputs = []
    for face in face_boxes_resized:
        print('Bounding box shape :', face.shape)
        face = cv2.resize(face, (160, 160))
        face = face.reshape(1, 160, 160, 3)
        face = prewhiten(face)
        print('Bounding box shape resized :', face.shape)
        output = model.predict(face)
        print('Output shape :', output[0].shape)
        outputs.append(output[0])
    return outputs

# need cudnn and CUDA toolkit installed for cuda as ONNXRuntime backend
detector_yolov5s6 = Detector('best_yolov5s6.onnx', 1/255.0, (640, 640), backend='cuda')
face_boxes = []
for image_location in os.listdir('test_images'):
    image = cv2.imread('test_images/' + image_location)
    # use yolov5s6 model to get bounding boxes of detected faces
    boxes = detector_yolov5s6.detect(image)
    # extract faces from image
    faces = [image[y:y+h,x:x+w] for x, y, w, h in boxes]
    face_boxes.extend(faces)

face_embeddings_casia = predict(face_boxes, 'FaceNet_CASIA_WebFace/FaceNet_CASIA_WebFace.h5')
face_embeddings_vgg = predict(face_boxes, 'FaceNet_VGGFace2/FaceNet_VGGFace2.h5')

# using DBSCAN on face embeddings
model = DBSCAN(eps=0.5, metric='cosine', min_samples=1)
faces_casia = model.fit_predict(face_embeddings_casia)
faces_vgg = model.fit_predict(face_embeddings_vgg)

faces_casia_dict = create_dictionary(faces_casia, face_boxes)
faces_vgg_dict = create_dictionary(faces_vgg, face_boxes)

# DBSCAN output
print('DBSCAN for FaceNet trained on CASIA_WebFace:', faces_casia)
print('DBSCAN for FaceNet trained on VGGFace2:', faces_vgg)

# save plot figures as png
plot_figure(faces_casia_dict, 'casia', True)
plot_figure(faces_vgg_dict, 'vgg', True)
