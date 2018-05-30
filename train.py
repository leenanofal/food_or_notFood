import numpy as np 
import json
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, RMSprop
from sklearn.metrics import classification_report
import imageloader
from imutils import paths
from simple_model import ShallowNet
from inception_resnet_v2 import LeenaNet
import imagepreprocessor as mp
from keras import backend as K



def food_or_not(label):
    if label == 'food':
        return 1.0
    else:
        return 0.0

#creating a dictionary that correlates the image_id with it's correct label
def image_label(infos):
    img_lab = {}
    for i in range(len(infos)):
        label = infos[i]['label']
        img_lab[infos[i]['photo_id']] = food_or_not(label)
    return img_lab

def pred_true_diff(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

def main():

    jsonFile = open('/home/Leena/dataset/photos/photos.json')
    infos = []
    #loop through the lines in the json file and append each one into the infos array
    imagePaths = list(paths.list_images('/home/Leena/dataset/photos/photos/'))
    for line in jsonFile:
        photo_info = json.loads(line) #gets each line in the json file
        infos.append(photo_info)
    
    #preprocess the images
    resizer = mp.ImageResize(299, 299)
    array_maker = mp.ImageToArrayPreprocessor()
    il = imageloader.SimpleDatasetLoader(preprocessors=[resizer, array_maker]) #building the image loader
    (images, img_ids) = il.load(imagePaths, max_images=1000, verbose=500)
    img_lab = image_label(infos)
    X = images #X is an array of all the images 
    Y = []     #corresponding labels for each image in X
    
    #add the labels into a list
    for image_id in img_ids:
        y = img_lab[image_id]
        Y.append(y)

    #split data into train and test sets 
    (trainX, testX, trainY, testY) = train_test_split(X, Y, test_size=0.25, random_state=42)
    #todo: add validation set

    print(Y[0])
    #initialize model
    print('COMPILING MODEL')
    #opt = SGD(lr=0.01)
    opt = RMSprop(lr=0.045, rho=0.9, epsilon=1.0, decay=0.9)
    #model = ShallowNet.build(width=299, height=299, depth=3, classes=1)
    model = LeenaNet.build(width=299, height=299, depth=3, classes=1)
    model.compile(loss="binary_crossentropy", optimizer=opt,
            metrics=['binary_accuracy','mean_absolute_error', pred_true_diff])

    # train the network
    print("TRAINING NETWORK!!!!")
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
            batch_size=32, epochs=4, verbose=1)

    # evaluate the network
    print("EVALUATING NETWORK...**fingers crossed**")
    predictions = model.predict(testX, batch_size=32)
    #print(classification_report(testY,
    #        predictions))


if __name__ ==  "__main__":
    main()
	

