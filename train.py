import numpy as np 
import json
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, RMSprop
from sklearn.metrics import classification_report
import imageloader
import argparse
from imutils import paths
from simple_model import ShallowNet
from inception_resnet_v2 import LeenaNet
import imagepreprocessor as mp
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from leena_logger import LeenaLogger


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default='/home/Leena/dataset/photos/', help="path to input dataset")
ap.add_argument("-e", "--evaluate", required=False, action='store_true', help="Only evaluate, no training")
ap.add_argument("-c", "--checkpoint", required=False, action='store_true', help="Saves checkpoints")
args = vars(ap.parse_args())


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

    # settings
    test_size = 0.25
    random_state = 42
    image_width = 299
    image_height = 299
    num_epochs = 30
    steps_per_epoch=5

    # hyperparameters to play with
    epoch_size = 1000 # maximum 206949
    learning_rate = 0.001
    momentum= 0.3 #0.5 #0.2
    decay = 0.001
    batch_size = 32
    # model parameters? 

    jsonFile = open(args["dataset"]+'photos.json')
    infos = []
    #loop through the lines in the json file and append each one into the infos array
    imagePaths = list(paths.list_images(args["dataset"]))
    for line in jsonFile:
        photo_info = json.loads(line) #gets each line in the json file
        infos.append(photo_info)
    
    #preprocess the images
    resizer = mp.ImageResize(299, 299)
    array_maker = mp.ImageToArrayPreprocessor()
    il = imageloader.SimpleDatasetLoader(preprocessors=[resizer, array_maker]) #building the image loader
    (images, img_ids) = il.load(imagePaths, max_images=epoch_size, verbose=500)
    img_lab = image_label(infos)
    X = images #X is an array of all the images 
    Y = []     #corresponding labels for each image in X
    
    #add the labels into a list
    for image_id in img_ids:
        y = img_lab[image_id]
        Y.append(y)

    #split data into train and test sets 
    (trainX, testX, trainY, testY) = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    #todo: add validation set

    print(Y[0])
    #initialize model
    print('COMPILING MODEL')
    opt = SGD(lr=learning_rate, momentum=momentum, decay=decay)
    #opt = RMSprop(lr=0.01, rho=0.9, epsilon=0.7, decay=0.9) #epsilon changed from 1
    #model = ShallowNet.build(width=299, height=299, depth=3, classes=1)
    model = LeenaNet.build(width=image_width, height=image_height, depth=3, classes=1)
    model.compile(loss="binary_crossentropy", optimizer=opt,
            metrics=['binary_accuracy','mean_absolute_error'])

    print(K.image_data_format())
    # train the network
    print("TRAINING NETWORK!!!!")
    tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=False)
    leenaLogger = LeenaLogger("log_epsize%d_lr%f_mom%f_dec%f_bs%d"%(epoch_size, learning_rate, momentum, decay, batch_size))
    callbacks = [tbCallBack, leenaLogger]
    if args["checkpoint"]:
        print("Saving checkpoints...")
        modelCheckpoint = ModelCheckpoint(filepath="models/model_epsize%d_lr%f_mom%f_dec%f_bs%d_{epoch:02d}_{val_loss:.2f}.hdf5"%(epoch_size, learning_rate, momentum, decay, batch_size))
        callbacks.append(modelCheckpoint)
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
            batch_size=batch_size, epochs=num_epochs, verbose=1, callbacks=callbacks, 
            steps_per_epoch=steps_per_epoch)
    
    
    # evaluate the network
    print("EVALUATING NETWORK...**fingers crossed**")
    predictions = model.predict(testX, batch_size=32)
    #print(classification_report(testY,
    #        predictions))


if __name__ ==  "__main__":
    main()
	

