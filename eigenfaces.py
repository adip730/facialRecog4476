# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:44:33 2018

@author: Luis
"""

import numpy as np
import matplotlib.pyplot as plt

class Eigenfaces:
    def __init__(self, dataset, variance):
        #These are the vectorized images (each row is an image)
        self.dataset = dataset
        self.n = dataset.shape[0]
        self.variance = variance

        # subtract mean
        self.mean_matrix = np.mean(dataset, axis=0)
        x = np.subtract(dataset, self.mean_matrix)

        # calculate covariance
        s = np.cov(np.transpose(x))

        # calculate eigenvalue & eigenvector
        # (already normalized)
        self.eigvals, self.eigvect = np.linalg.eig(s)

        # sort eigenvalues in descending order
        eigvals_index = self.eigvals.argsort()[::-1]
        self.eigvals = self.eigvals[eigvals_index]
        self.eigvect = self.eigvect[eigvals_index].real

        # evaluate the number of principal components needed
        # to represent "variance" Total variance
        eigsum = np.sum(self.eigvals);
        csum = 0;
        for i in range(self.n):
            csum += self.eigvals[i]
            tv = csum / eigsum
            if tv > self.variance:
                variance_index = i
                break

        # select those above variance
        self.eigvals = self.eigvals[:variance_index]
        self.eigvect = self.eigvect[:,:variance_index]

        # compute weights for each img using eigvect
        weights = []
        for i in range(self.n):
            weights.append(self.eigvect.transpose() * x[i,:])
        self.weights = np.array(weights)


    def recognition(self, img):
        vect_img = img.astype(np.float).flatten()
        vect_img -= self.mean_matrix

        projection = self.eigvect.transpose() * vect_img

        diff = projection - self.weights[0]
        norms = np.linalg.norm(diff, axis=1)
        idx = np.argmin(norms)
        min_diff = norms[idx]
        min_idx = 0
        for i in range(1, self.n):
            diff = projection - self.weights[i]
            norms = np.linalg.norm(diff, axis=1)
            idx = np.argmin(norms)

            if norms[idx] < min_diff:
                min_diff = norms[idx]
                min_idx = i

        # Always returning the closest result, no threshold applied
        return min_idx


    def detection(self, img, threshold):
        face = False
        vect_img = img.astype(np.float).flatten()
        vect_img -= self.mean_matrix

        projection = self.eigvect.transpose() * vect_img

        diff = vect_img - projection
        norms = np.linalg.norm(diff, axis=1)

        if np.amin(norms) < threshold:
            face = True

        return face

################################ TESTING ################################
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from skimage.transform import resize
from rgb2gray import rgb2gray


dataset = fetch_lfw_people(min_faces_per_person=100)
h = dataset.images.shape[1]
w = dataset.images.shape[2]

X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# Create and call eigenfaces
# TODO: Studying and finding a suitable general variance
variance = 0.95
eigenfaces = Eigenfaces(X_train, variance)

# Showing the input image
input_img_id = 240
plt.subplot(121)
plt.title("Input image")
plt.imshow(X_test[input_img_id].reshape(h,w), plt.get_cmap('gray'))

# Showing the image that the input image was indentified as
plt.subplot(122)
plt.title("Identified image")
face_id = eigenfaces.recognition(X_test[input_img_id])
plt.imshow(X_train[face_id].reshape(h,w), plt.get_cmap('gray'))
plt.show()

# Detecting faces (this should tell you whether the input image is a face or not)
# TODO: Working on the appropiate threshold value
img = plt.imread("jupiter.jpg", format='jpg')
img = rgb2gray(img)
img = resize(img, (h,w), mode='reflect')

threshold = 3000
print(eigenfaces.detection(img, threshold))
