import numpy as np
import matplotlib.pyplot as plt

class Fisherfaces:
    def __init__(self, dataset, variance, classes, y_train):
        #vectorized images
        self.dataset = dataset
        self.n = dataset.shape[0]

        self.variance = variance

        #pca
        self.mean_matrix = np.mean(dataset, axis=0)
        x = np.subtract(dataset, self.mean_matrix)

        s = np.cov(np.transpose(x))

        self.eigenvalues, self.eigenvectors = np.linalg.eig(s)

        eigenvalues_index = self.eigenvalues.argsort()[::-1]
        self.eigenvalues = self.eigenvalues[eigenvalues_index]
        self.eigenvectors = self.eigenvectors[eigenvalues_index].real

        eig_sum = np.sum(self.eigenvalues)
        c = 0
        for i in range(self.n):
            c += self.eigenvalues[i]
            tv = c/eig_sum
            if tv > self.variance:
                variance_index = i
                break

        self.eigenvalues = self.eigenvalues[:variance_index]
        self.eigenvectors = self.eigenvectors[:,:variance_index]


        #lda
        y = y_train
        self.classes = classes
        X = np.dot(self.dataset-self.mean_matrix, self.eigenvectors)
        [n,d] = X.shape
        meanTotal = X.mean(axis=0)
        s_b = np.zeros((d,d), dtype=np.float32)
        s_w = s_b
        
        for i in self.classes:
            x_i = X[np.where(y == i)[0],:]
            meanClass = x_i.mean(axis=0)
            s_b += n*np.dot((meanClass - meanTotal).transpose(),(meanClass - meanTotal))
            s_w += np.dot((x_i - meanClass).transpose(), (x_i - meanClass))
        self.eigenvalues_lda, self.eigenvectors_lda = np.linalg.eig(np.linalg.inv(s_w)*s_b)

        eigenvalues_index = np.argsort(-self.eigenvalues_lda.real) 
        self.eigenvalues_lda = self.eigenvalues_lda[eigenvalues_index]
        self.eigenvectors_lda = self.eigenvectors_lda[:,eigenvalues_index]

        self.eigenvalues_lda = np.array(self.eigenvalues_lda[0:len(self.classes)-1].real, dtype=np.float32, copy=True)
        self.eigenvectors_lda = np.array(self.eigenvectors_lda[0:,0:len(self.classes)-1].real, dtype=np.float32, copy=True)
        
        self.eigenvectors = np.dot(self.eigenvectors, self.eigenvectors_lda)

        weights = []
        for i in range(self.n):
            weights.append(self.eigenvectors.transpose() * x[i, :])
        self.weights = np.array(weights)


    def recognition(self, img):
        vect_img = img.astype(np.float32).flatten()
        vect_img -= self.mean_matrix

        projection = self.eigenvectors.transpose() * vect_img

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

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from skimage.transform import resize

def main(label):
    dataset = fetch_lfw_people(min_faces_per_person=100)
    h = dataset.images.shape[1]
    w = dataset.images.shape[2]

    data = dataset.data
    target = dataset.target
    classes = np.unique(target)

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

    variance = 0.95
    fisherfaces = Fisherfaces(x_train, variance, classes, y_train)

    #show input image
    input_img_id = 240
    counter = 0
    for i in range(len(y_test)):
        if y_test[i] == label:
            if counter == 5:
                input_img_id = i
                break
            counter = counter + 1
        
    #plt.subplot(121)
    #plt.title("input")
    #plt.imshow(x_test[input_img_id].reshape(h,w), plt.get_cmap('gray'))

    #show predicted
    #plt.subplot(122)
    #plt.title("predicted")
    face_id = fisherfaces.recognition(x_test[input_img_id])
    #plt.imshow(x_train[face_id].reshape(h,w), plt.get_cmap('gray'))
    #plt.show()
    #input person
    input_person = dataset.target_names[y_test[input_img_id]]
    #print("Input:")
    #print(input_person)
    #predicted person
    pred_person = dataset.target_names[y_train[face_id]]
    #print("\nPredicted:")
    #print(pred_person)

    if input_person == pred_person:
        return(1, input_person)
    else:
        return(0, pred_person)

#number of times to run the loop
num_trials= 25
#initial score
score = 0
#input image id
initial_id = 0
for i in range(num_trials):
    x = main(initial_id)
    score = score + x[0]
    print(x)
    
print(score)
total_accuracy = score/num_trials
print(total_accuracy)

               
