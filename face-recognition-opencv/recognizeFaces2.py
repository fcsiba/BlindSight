# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 

# import the necessary packages
from __main__ import *
import face_recognition
import argparse
import pickle
import cv2
import sys
from gtts import gTTS 
import urllib.request
import requests
import base64

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-e", "--encodings", required=False,
#	help="path to serialized db of facial encodings")
#ap.add_argument("-i", "--image", required=False,
#	help="path to input image")
#ap.add_argument("-d", "--detection-method", type=str, default="hog",
#	help="face detection model to use: either `hog` or `cnn`")
#args = vars(ap.parse_args())
language = 'en'
print("inner module starting");
args = {}
args["encodings"] = "encodings.pickle"
args["detection_method"] = "hog"
# load the known faces and embeddings
#print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

files=os.listdir('in/')[1:]


args["image"] = files[0]
# load the input image and convert it from BGR to RGB
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
#print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb,
	model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# initialize the list of names for each face detected
names = []

# loop over the facial embeddings
unknowns = []
s = 1
for encoding in encodings:
	# attempt to match each face in the input image to our known
	# encodings
	matches = face_recognition.compare_faces(data["encodings"],
		encoding)
	name = "Unknown"+str(s)
	# check to see if we have found a match
	if True in matches:
		# find the indexes of all matched faces then initialize a
		# dictionary to count the total number of times each face
		# was matched
		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		counts = {}

		# loop over the matched indexes and maintain a count for
		# each recognized face face
		for i in matchedIdxs:
			name = data["names"][i]
			counts[name] = counts.get(name, 0) + 1

		# determine the recognized face with the largest number of
		# votes (note: in the event of an unlikely tie Python will
		# select first entry in the dictionary)
		name = max(counts, key=counts.get)
	else:
        s = s + 1
        unknowns.append(name)
	# update the list of names

	if '/' in name:
		name = name.rsplit('/', 1)[-1]
	names.append(name)
# loop over the recognized faces
print(names)
print(unknowns)
#sys.exit()
#for ((top, right, bottom, left), name) in zip(boxes, names):
#	# draw the predicted face name on the image
#	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
#	y = top - 15 if top - 15 > 15 else top + 15
#	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
#		0.75, (0, 255, 0), 2)

text = ""
for a in names:
	text = text + a + ", "

 
if text != "":
 	myobj = gTTS(text=text+" is around you!", lang=language, slow=False) 
 	myobj.save(temp+".mp3") 

url = "http://blindsight.000webhostapp.com/SaveAudio.php"
myFile = open("F:/FYP/face-recognition-opencv/face-recognition-opencv/"+temp+".mp3","rb")
#myFile = myFile.encode("utf-8")
encodedA = base64.b64encode(myFile.read())
r = requests.post(url,data={'name':temp,'image':encodedA},files={'file':myFile})



# show the output image
#cv2.imshow("Image", image)
#cv2.waitKey(0)
