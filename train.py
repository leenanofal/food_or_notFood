import numpy as np 
import json



def main():
    jsonFile = open('/home/Leena/dataset/photos/photos.json')
    infos = []
    #loop through the lines in the json file and append each one into the infos array
    for line in jsonFile:
        photo_info = json.loads(line) #gets each line in the json file
        infos.append(photo_info)


    for i in range(10):
        print(infos[i]['label']) #json format allows you to do this: gets the label value from each line


if __name__ ==  "__main__":
    main()
	

