import os
import numpy as np
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from detector import Detector
from facenet import FaceNet
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
        file_dir = 'assets/'+ model_name +'.png'
        print('Saving ' + model_name.upper() + ' dataset results at ' + file_dir)
        plt.savefig(file_dir)
    else:
        plt.show()

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

def predict(faces, model_name, backend):
    outputs = []
    model = FaceNet(model_name, 1/255.0, (160, 160), backend)
    for face in faces:
        print('Bounding box shape :', face.shape)
        output = model.generate_embeddings(face)
        print('Output shape :', output.shape)
        outputs.append(output)
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

# set backend from 'tf', 'cuda' and 'onnx'
backend = 'cuda'
model_casia = 'FaceNet_CASIA_WebFace/FaceNet_CASIA_WebFace.onnx'
model_vgg = 'FaceNet_VGGFace2/FaceNet_VGGFace2.onnx'
face_embeddings_casia = predict(face_boxes, model_casia, backend)
face_embeddings_vgg = predict(face_boxes, model_vgg, backend)

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
