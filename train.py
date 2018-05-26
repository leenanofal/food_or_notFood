import numpy as np 
import json
import imageloader
from imutils import paths

def main():
    jsonFile = open('/home/Leena/dataset/photos/photos.json')
    infos = []
    #loop through the lines in the json file and append each one into the infos array
    imagePaths = list(paths.list_images('/home/Leena/dataset/photos/photos/'))
    for line in jsonFile:
        photo_info = json.loads(line) #gets each line in the json file
        infos.append(photo_info)
    il = imageloader.SimpleDatasetLoader(preprocessors=[]) #building the image loader
    (images, img_ids) = il.load(imagePaths, max_images=1000, verbose=500)
    

    for i in range(10):
        print(infos[i]['label']) #json format allows you to do this: gets the label value from each line
        print(img_ids[i])

if __name__ ==  "__main__":
    main()
	

