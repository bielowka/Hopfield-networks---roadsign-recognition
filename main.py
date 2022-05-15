from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import numpy as np
import os


SIZE = 100
STEPS = 10

DIRECTORY = "znakismallimproved"
classes = ['maszpierwszenstwo','rondo','stop','ustap','zakazzatrzymywania']
classesToNames = {'M': 'maszpierwszenstwo',
               'R': 'rondo',
               'S': 'stop',
               'U': 'ustap',
               'Z': 'zakazzatrzymywania'}

namesToClasses = {'maszpierwszenstwo': 'M',
               'rondo': 'R',
               'stop': 'S',
               'ustap': 'U',
               'zakazzatrzymywania': 'Z'}

NUM_OF_CLASSES = len(classes)


def preprocess(path):
    img_path = os.path.join(path)
    image = load_img(img_path, target_size=(SIZE, SIZE))
    image = img_to_array(image)
    image = preprocess_input(image)
    data = []
    data.append(image)
    data = np.array(data,dtype="float32")
    return data

def imgBinarized(data): #changes image to black and white only
    dataT = []
    for i in data:
        tmp = []
        for j in i:
            line = []
            for k in j:
                sum = 0
                for l in k: sum += l
                if sum > 255 * 3 / 2:
                    tmp.append(-1)
                else:
                    tmp.append(1)
        dataT.append(np.array(tmp))

    return dataT


def train(neu, training_data):
    w = np.zeros([neu, neu])
    for data in training_data:
        w += np.outer(data, data)
    for diag in range(neu):
        w[diag][diag] = 0
    return w

def test(W, test_data):
    success = 0.0
    output_data = []
    for data in test_data:
        true_data = data[0]
        noisy_data = data[1]
        predicted_data = retrieve_pattern(W, noisy_data)
        success += ((SIZE*SIZE - np.sum(abs(true_data - predicted_data)))/(SIZE*SIZE))*100 #number of pixels got right
        output_data.append([true_data, noisy_data, predicted_data])

    return (success / (len(test_data)*100)), output_data

def retrieve_pattern(weights, data, steps=STEPS):
    res = np.array(data)
    for _ in range(steps):
        for i in range(len(res)):
            raw_v = np.dot(weights[i], res)
            if raw_v > 0:
                res[i] = 1
            else:
                res[i] = -1
    return res

def plot_images(images, title, no_i_x=10, no_i_y=3):
    fig = plt.figure(figsize=(10, 15))
    fig.canvas.set_window_title(title)
    images = np.array(images).reshape(-1, SIZE, SIZE)
    images = np.pad(images, ((0, 0), (1, 1), (1, 1)), 'constant', constant_values=-1)
    for i in range(no_i_x):
        for j in range(no_i_y):
            ax = fig.add_subplot(no_i_x, no_i_y, no_i_x * j + (i + 1))
            ax.matshow(images[no_i_x * j + i], cmap="gray")
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

            if j == 0 and i == 0:
                ax.set_title("Real")
            elif j == 0 and i == 1:
                ax.set_title("Distorted")
            elif j == 0 and i == 2:
                ax.set_title("Reconstructed")

data = []
labels = []


for znak in os.listdir(DIRECTORY):
    path = os.path.join(DIRECTORY,znak)
    for img in os.listdir(path):
        img_path = os.path.join(path,img)
        image = load_img(img_path,target_size=(SIZE, SIZE))
        image = img_to_array(image)

        data.append(image)
        labels.append(namesToClasses[znak])


data = np.array(data,dtype="float32")
data = imgBinarized(data)


n_side = SIZE
n_neurons = n_side * n_side


percent = 5
# percent = 7
# percent = 8
# percent = 12

train_data = [data[percent*i] for i in range(len(data)//percent)]
train_labels = [labels[percent*i] for i in range(len(labels)//percent)]
print(len(train_data))

n_test = len(train_data)
distort = 0.1

test_data = []

for d in range(n_test):
    base_pattern = np.array(train_data[d])
    noise = 1 * (np.random.random(base_pattern.shape) > distort)
    np.place(noise, noise == 0, -1)
    noisy_pattern = np.multiply(base_pattern, noise)
    test_data.append((base_pattern, noisy_pattern))

W = train(n_neurons, train_data)

accuracy, op_imgs = test(W, test_data)

print("Accuracy of the network is %f" % (accuracy * 100))
n_train_disp = min(len(train_labels),10)
plot_images(op_imgs, "Reconstructed Data", n_train_disp)
plt.show()
