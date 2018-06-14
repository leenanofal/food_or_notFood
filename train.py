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
from keras.models import load_model
import batchloader
import time

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default='/home/Leena/dataset/photos/', help="path to input dataset")
ap.add_argument("-e", "--evaluate", required=False, action='store_true', help="Only evaluate, no training")
ap.add_argument("-t", "--testdata", required=False, action='store_true', help="Only loads test data for evaluation")
ap.add_argument("-c", "--checkpoint", required=False, action='store_true', help="Saves checkpoints")
ap.add_argument("-m", "--model", required=False, help="Path to load model")
args = vars(ap.parse_args())


def food_or_not(label):
    if label == 'food':
        return 1.0
    else:
        return 0.0

def confidence_to_food_or_not(y):
    if y > 0.5:
        return ('food', (y))
    else:
        return ('notFood', (1-y))

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
    test_size = 0.02
    random_state = 42
    image_width = 299
    image_height = 299
    num_epochs = 2
    #steps_per_epoch=5
    epoch_split = 20

    # hyperparameters to play with
    epoch_size = 140000 # maximum 206949
    learning_rate = 0.001
    momentum= 0.3 #0.5 #0.2
    decay = 0.001
    batch_size = 32
   

    if (epoch_size * (1-test_size)) % epoch_split != 0:
        print("ERROR: make sure even split")
        exit()

    
    #preprocess the images
    imagePaths = list(paths.list_images(args["dataset"]))
    resizer = mp.ImageResize(299, 299)
    array_maker = mp.ImageToArrayPreprocessor()
    il = imageloader.SimpleDatasetLoader(preprocessors=[resizer, array_maker]) #building the image loader

    #print(imagePaths)
    if not args['testdata']:
        if args['evaluate']:
            (images, img_ids) = il.load(imagePaths, max_images=epoch_size, verbose=500)
        else:
            (images, img_ids) = batchloader.parallel_load(il, imagePaths, numImages=epoch_size, batchSize=500, numThreads=20)
        
        X = images #X is an array of all the images 
    
    # model building/restoring
    if args['model']:
        model = load_model(args['model'])
    else:
        #initialize model
        print('COMPILING MODEL')
        opt = SGD(lr=learning_rate, momentum=momentum, decay=decay)
        #opt = RMSprop(lr=0.01, rho=0.9, epsilon=0.7, decay=0.9) #epsilon changed from 1
        #model = ShallowNet.build(width=299, height=299, depth=3, classes=1)
        model = LeenaNet.build(width=image_width, height=image_height, depth=3, classes=1)
        model.compile(loss="binary_crossentropy", optimizer=opt,
                metrics=['binary_accuracy','mean_absolute_error'])
   
    #Running model on any images user inputs 
    if args['evaluate']:
        print("evaluating performance...")
        # TODO: image evaluation
        predictions = model.predict(X)
        for i in range(len(X)):
            imgPath = imagePaths[i]
            pred = confidence_to_food_or_not(predictions[i])
            print("Prediction: %s ; Confidence %f; Image: %s" % (pred[0], pred[1], imgPath))
    

    #evaluating the test data
    #model has never seen test images before evaluation
    elif args['testdata']:
        print("Running test set...")
        # TODO: image evaluation
        begin = 140000
        num_test = 2000
        (images, img_ids) = batchloader.parallel_load(il, imagePaths[begin:begin+num_test], numImages=num_test, batchSize=100, numThreads=20)
        X = images #X is an array of all the images 
        
        jsonFile = open(args["dataset"]+'photos.json')
        infos = []
        #loop through the lines in the json file and append each one into the infos array
        for line in jsonFile:
            photo_info = json.loads(line) #gets each line in the json file
            infos.append(photo_info)
        
        img_lab = image_label(infos)
        Y = []     #corresponding labels for each image in X
        
        #add the labels into a list
        for image_id in img_ids:
            y = img_lab[image_id]
            Y.append(y)
        predictions = model.predict(X, verbose=1)

        
        
        ##METRICS
        
        #Accuracy --- the fraction of predictions that the model got correct. 
        rounded_predictions = np.squeeze(np.rint(predictions)) # rounds the predictions to the nearest int -- i.e. 1 or 0
        print(rounded_predictions)
        rounded_truth = np.rint(Y) # rounds the label values because they are floats
        print(rounded_truth)
        tru_pos = np.sum((rounded_truth == 1) & (rounded_predictions == 1))
        print(tru_pos)
        tru_neg = np.sum((rounded_truth == 0) & (rounded_predictions == 0))
        num_correct = tru_pos + tru_neg
        total_pics = num_test
        acc = num_correct/total_pics
        print("Accuracy: " , acc)

        #Precision = true positive rate / (true positive rate + false positive rate)---identifies the proportion 
        #   of the positives identified were actually positive.
        false_pos = np.sum((rounded_truth == 0) & (rounded_predictions == 1))
        precision = tru_pos / (tru_pos + false_pos)
        print("Precision: " , precision)
        
        #Recall = true positive rate / (true positive rate + false negative rate) --- identifies how much 
        #   of the actual positives were correctly identified 
        false_neg = np.sum((rounded_truth == 1) & (rounded_predictions == 0))
        recall = tru_pos / (tru_pos + false_neg)
        print("Recall: " , recall)

        #Specificity --- the percentative of negatives we predict correctly. 
        specificity = tru_neg/(tru_neg + false_pos)
        print("Specificity" , specificity)
        
        logname = "./logs/" + "test_results" + "_" + time.strftime("%Y%m%d_%H%M%S") + ".csv"
        f = open(logname, 'w')
        
        f.write("image,prediction,truth,correct,classification,confidence")
        f.write("\n")
        for i in range(total_pics):
            predi = confidence_to_food_or_not(predictions[i])
            f.write("%s,%d,%d,%d,%s,%f"%(img_ids[i],int(rounded_predictions[i]), int(rounded_truth[i]), int(rounded_predictions[i] == rounded_truth[i]),predi[0],predi[1]))
            f.write("\n")

        f.close()


    else:
        # train, not evaluate

        jsonFile = open(args["dataset"]+'photos.json')
        infos = []
        #loop through the lines in the json file and append each one into the infos array
        for line in jsonFile:
            photo_info = json.loads(line) #gets each line in the json file
            infos.append(photo_info)
        
        img_lab = image_label(infos)
        Y = []     #corresponding labels for each image in X
        
        #add the labels into a list
        for image_id in img_ids:
            y = img_lab[image_id]
            Y.append(y)

        #split data into train and test sets 
        (trainX, testX, trainY, testY) = train_test_split(X, Y, test_size=test_size, random_state=random_state)
        #todo: add validation set

        print(Y[0])

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
        
        #splitting up the epochs so that I don't train on 200k images in one epoch
        #training on 10k at a time
        for ep_num in range(num_epochs):
            for semi_ep_num in range(epoch_split):
                start_idx = int(semi_ep_num*(len(trainX)/epoch_split))
                end_idx = int((semi_ep_num+1)*(len(trainX)/epoch_split))
                print("Epoch %d, Semi-epoch %d - idx[%d:%d]"%(ep_num, semi_ep_num, start_idx, end_idx))
                semi_trainX = trainX[start_idx:end_idx]
                semi_trainY = trainY[start_idx:end_idx]
                
                H = model.fit(semi_trainX, semi_trainY, validation_data=(testX, testY),
                        batch_size=batch_size, epochs=(ep_num+1), initial_epoch=ep_num, verbose=1, callbacks=callbacks) 

        
        #    H = model.fit(trainX, trainY, validation_data=(testX, testY),
        #            batch_size=batch_size, epochs=num_epochs, verbose=1, callbacks=callbacks, 
        #            steps_per_epoch=steps_per_epoch)
            
        
        # evaluate the network
        print("EVALUATING NETWORK...**fingers crossed**")
        predictions = model.predict(testX, batch_size=32)
        #print(classification_report(testY,
        #        predictions))


if __name__ ==  "__main__":
    main()
	

