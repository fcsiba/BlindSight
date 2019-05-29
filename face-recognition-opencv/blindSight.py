#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 07:25:48 2019

@author: rehan
"""
import re   

from bs4 import BeautifulSoup
import requests
from sys import byteorder
from array import array
from struct import pack
import pyaudio
import wave
from gtts import gTTS 
from pygame import mixer # Load the required library
import threading
from imageai.Detection import ObjectDetection
import os
import collections
from PIL import Image
import pytesseract
import argparse
import cv2
import os
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
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

import base64
import requests

import cv2
import numpy as np
from keras.models import load_model
import sys
import warnings

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


from pydub import AudioSegment
from pydub.playback import play



os.chdir('/Users/rehan/Documents/FYP/PROJECT/face-recognition-opencv')

pytesseract.pytesseract.tesseract_cmd = r"/usr/local/Cellar/tesseract/4.0.0_1/bin/tesseract"


ansList = []


##Satart Section
''' Keras took all GPU memory so to limit GPU usage, I have add those lines'''


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))
''' Keras took all GPU memory so to limit GPU usage, I have add those lines'''
## End section


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
model = load_model('model_5-49-0.62.hdf5')

#OBJECT

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( "/Users/rehan/Documents/FYP/PROJECT/resnet50_coco_best_v2.0.1.h5")

detector.loadModel()




#os.getcwd()
#os.chdir('face-recognition-opencv')









def faceRecognition(imagePath):
    language = 'en'
    print("inner module starting");
    args = {}
    args["encodings"] = "encodings.pickle"
    args["detection_method"] = "hog"
    # load the known faces and embeddings
    #print("[INFO] loading encodings...")
    data = pickle.loads(open(args["encodings"], "rb").read())
    
    
    
    args["image"] = imagePath
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
    for encoding in encodings:
    	# attempt to match each face in the input image to our known
    	# encodings
    	matches = face_recognition.compare_faces(data["encodings"],
    		encoding)
    	name = "Unknown"
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
    	
    	# update the list of names
    	if '/' in name:
    		name = name.rsplit('/', 1)[-1]
    	names.append(name)
    # loop over the recognized faces
    
    if (len(names)==0):
        return('We could\'t find any faces')
    else:
        retAns = "We detected "+' and '.join(names)
        
        return(retAns)
    
    #return(names)
    #sys.exit()
    #for ((top, right, bottom, left), name) in zip(boxes, names):
    #	# draw the predicted face name on the image
    #	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    #	y = top - 15 if top - 15 > 15 else top + 15
    #	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
    #		0.75, (0, 255, 0), 2)
    
    
    
    
    
    # show the output image
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
def retAns(mylist):
    counter=collections.Counter(mylist)
    
    noOb = len(mylist)

    if noOb > 0:
        
        if noOb>1:
            ans= 'There are a total of '+str(noOb)+' objects in this picture.\n'
        else:
            return('There is just one object in this picture. A '+mylist[0]+'.\n')
    
    
    
        if len(counter.keys())==1:
            ans = ans+'All of those are '+list(counter.keys())[0]+'s.\n'
            return(ans)
        for i in counter.keys():
            if counter[i]>1:
                
                ans=ans+str(counter[i])+' of those are '+i+'.\n'
            else:
                ans=ans+str(counter[i])+' of those is a '+i+'.\n'
    
    else:
        ans = 'We couldn\'t detect any objects in this picture. Please try again!'
        
    return(ans)
    
    
def objectRecognition(imagePath,detector):
    execution_path = os.getcwd()
    #custom_objects = detector.CustomObjects(person=True)
    
    results={}
    ob=[]
    obPercent=[]
    #detections, extracted_images = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "in/"+f), output_image_path=os.path.join(execution_path , "out/"+f), extract_detected_objects=True)
    
    obList=[]
    
    detections = detector.detectObjectsFromImage(input_image=imagePath, output_image_path=os.path.join(execution_path , "output/objectRecognition.JPG"))
    ob=[]
    obPercent=[]
    for eachObject in detections:
        #results[]=eachObject["percentage_probability"]
        
        ob.append(eachObject["name"])
        obPercent.append(eachObject["percentage_probability"])
    obList.append(ob)
    return(retAns(ob))
    
        
def textRecognition(imagePath):
    

    os.system('convert '+imagePath+' -resize 400% -type Grayscale output/text.tif')   
    
    
    return(pytesseract.image_to_string(Image.open('output/text.tif')))
    #return(pytesseract.image_to_string(Image.open('output/text.tif')))
    
import speech_recognition as sr
r = sr.Recognizer()


#r.recognize_google()
count = 0
def captureImage():
    global count
    import cv2
    video_capture = cv2.VideoCapture(0)
    # Check success
    if not video_capture.isOpened():
        raise Exception("Could not open video device")
    # Read picture. ret === True on success
    ret, frame = video_capture.read()
    
    fName = str(count)+'.jpg'
    cv2.imwrite('output/'+fName,frame)
    count=count+1
    # Close device
    video_capture.release()
    execution_path = os.getcwd()
    
    return(execution_path+'/output/'+fName)
    
    
  




def saveData(userName):
    language = 'en'
    cascPath = 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)
    people = []
    saving = False
    temp_path = ""
    name = ""
    video_capture = cv2.VideoCapture(0)
    i = 0
    j = 0
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
           # minSize=(30, 30)
           # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        #path = 'F:/FYP/face-recognition-opencv/face-recognition-opencv/unknown'
        # Draw a rectangle around the faces
        #if not os.path.exists('F:/FYP/face-recognition-opencv/face-recognition-opencv/unknown'):
         #   os.makedirs('F:/FYP/face-recognition-opencv/face-recognition-opencv/unknown')
       # print("what")
        for (x, y, w, h) in faces:
          #  print("fa")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            crop_img = frame[y:y+h, x:x+w].copy()
            imNameCopy = "{0}".format(i)
            i = i + 1
            if not saving:
     #           cv2.imwrite(os.path.join(path , imNameCopy+".jpg"), crop_img)
     #           import recognize_faces_image
      #          if 'Unknown' in recognize_faces_image.names or recognize_faces_image.names==[]:
       #             print("Give unknown a name!")
        #            name = input("Give a name")
                    temp_path = '/Users/rehan/Documents/FYP/PROJECT/face-recognition-opencv/profiles/'+userName
                    if not os.path.exists(temp_path):
                        os.makedirs(temp_path)
                    #else:
                     #   temp_len = len(os.listdir('F:/FYP/face-recognition-opencv/face-recognition-opencv/profiles/'))
                      #  print("Profile with this name already exists!")
                       # os.makedirs(temp_path+str(temp_len))
                        #print("Profile made instead with name"+name+str(temp_len))
                        #temp_path = 'F:/FYP/face-recognition-opencv/face-recognition-opencv/profiles/'+name +str(temp_len)
                    
                    saving = True
     #           else:
      #              for a in recognize_faces_image.names:
       #                 myobj = gTTS(text=a+" is around you!", lang=language, slow=False) 
        #                myobj.save("welcome.mp3") 
      
                        # Playing the converted file 
         #               playsound("welcome.mp3")
          #              sys.exit() 
            else:
                cv2.imwrite(os.path.join(temp_path , imNameCopy+".jpg"), crop_img)
                temp_len = len(os.listdir(temp_path))
                temp_len = temp_len + 1
                cv2.imwrite(os.path.join(temp_path , name+str(temp_len)+".jpg"), crop_img)
                j = j + 1
                if j>20:
                    #import encode_faces
                    saving = False
                    video_capture.release()
                    return




    
def encodeModel():
    import encode_faces
    
def test_image(addr):
    target = ['angry','disgust','fear','happy','sad','surprise','neutral']
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    im = cv2.imread(addr)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1)
    resultExp=[]
    for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2,5)
            face_crop = im[y:y+h,x:x+w]
            face_crop = cv2.resize(face_crop,(48,48))
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            face_crop = face_crop.astype('float32')/255
            face_crop = np.asarray(face_crop)
            face_crop = face_crop.reshape(1, 1,face_crop.shape[0],face_crop.shape[1])
            result = target[np.argmax(model.predict(face_crop))]
            resultExp.append(result)
            cv2.putText(im,result,(x,y), font, 1, (200,0,0), 3, cv2.LINE_AA)
            
    #cv2.imshow('result', im)
    cv2.imwrite('output/result_emotion_detection_app.jpg',im) 
    outP="No emotions detected"
    if(len(resultExp)>0):
        outP=""
        counter=collections.Counter(resultExp)
        for ci in list(counter.keys()):
            if(counter[ci]>1):
                outP=outP+str(counter[ci])+" are "+ci+". "
            else:
                outP=outP+str(counter[ci])+" is "+ci+". " 
    return(outP)
    
def recordAudio(mpname,s):
    
    import pyaudio
    import wave
    
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 44100  # Record at 44100 samples per second
    seconds = s
    filename = mpname
    
    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    
    print('Recording')
    
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)
    
    frames = []  # Initialize array to store frames
    
    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    
    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()
    
    print('Finished recording')
    
    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    AUDIO_FILE = mpname
    # use the audio file as the audio source
    try:
        
        with sr.AudioFile(AUDIO_FILE) as source:
            audio = r.record(source)  # read the entire audio file
    
        ans=r.recognize_google(audio)
    except:
        ans=''
    if 'break' ==ans:
        main()
    return(ans)
    
    
r = sr.Recognizer()



def sayText(mytext,fName):
    myobj = gTTS(text=mytext, lang='en', slow=False) 
  
    myobj.save(fName) 
          
        # Playing the converted file 
        
    song = AudioSegment.from_mp3(fName)
    play(song)
        



AUDIO_FILE = "demo.wav"




noteCount = 0
def noteTaking():
    global noteCount
    song = AudioSegment.from_mp3("startRecording.mp3")
    play(song)
    
    note = recordAudio('notes/'+str(noteCount)+'.wav',3)
    
    song = AudioSegment.from_mp3("endRecording.mp3")
    play(song)
    
    song = AudioSegment.from_mp3("nameRecording.mp3")
    play(song)
    
    #name = recordAudio('notes/'+str(noteCount)+'.wav',5)
    
    while(True):
        name = recordAudio('notes/'+str(noteCount)+'.wav',5)
        print(name)
        try:
            f=open('notes/'+str(name)+'.txt','w+')
            f.write(note)
            f.close()
            noteCount= noteCount +1
            break
        except:
            song = AudioSegment.from_mp3("errorRecording.mp3")
            play(song)
    
def readArticle():
    url = "https://www.dawn.com/"

# Getting the webpage, creating a Response object.
    response = requests.get(url)
    
    # Extracting the source code of the page.
    data = response.text
    soup = BeautifulSoup(data, 'html')
    mydivs = soup.findAll("h2", {"class": "story__title"})
    
    result=""
    
    for tArticle in range(0,5):
        result=result+ str(mydivs[tArticle].text)+". "
        
    
    
    sayText(result,'newsArticle.mp3')

    


url = "https://www.tennisexpress.com/search.cfm?searchKeyword=BB6892"

# Getting the webpage, creating a Response object.
response = requests.get(url)

# Extracting the source code of the page.
data = response.text

    
    
def noteSearch():
    
    allFiles=os.listdir('notes/')
    
    song = AudioSegment.from_mp3("searchRecording.mp3")
    play(song)
    
    name = recordAudio('tempQuery.wav',5)
    
    for t in allFiles:
        if name in t:
            song = AudioSegment.from_mp3("foundRecording.mp3")
            play(song)
            
            fR=open('notes/'+t,'r')
            readFile=fR.read()
            
            sayText(readFile,'tempQuery.mp3')
            return
    song = AudioSegment.from_mp3("notFoundRecording.mp3")
    play(song)
    
    
    
    
    
    



def main(speak = True):
    
    while(True):
        print("please speak a word into the microphone")

        if(speak):
            song = AudioSegment.from_mp3("homeRecording.mp3")
            play(song)
            speak=False
        
        
        ans = recordAudio("demo.wav",5)
        print(ans)
        
        if('detect' in ans or 'recognize' in ans or 'face' in ans or 'object' in ans or 'objects' in ans or 'text' in ans or 'read' in ans):
            imagePath = captureImage()
            #imagePath='/Users/rehan/Documents/FYP/PROJECT/face-recognition-opencv/2019_05_27_13_24_34.JPG'
            if('face' in ans or 'faces' in ans):
                output=faceRecognition(imagePath)
                output=output+". "+test_image(imagePath)
    
                
            elif('objects' in ans or 'object' in ans):
                output=objectRecognition(imagePath,detector)
            elif('text' in ans or 'read' in ans):
                output = textRecognition(imagePath)
                output=output.replace('\n',' ')
                if (len(output)>100):
                    output=output[0:500]
                if(len(output)==0):
                    output='No text recognized'
            else:
                output='No output'
                
            sayText(output,'welcome.mp3')
            print(output)
        elif('save' in ans or 'store' in ans):
            
            asp=ans.split(' ')
            if((len(asp)>1)):
                saveUser = 'Saving user as '+asp[1]+'. Please make sure there is only the user in the camera frame'
                
                
                sayText(saveUser,'sUser.mp3')
                
                
            
                saveData(asp[1])
                
                sayText('User Saved','sUser2.mp3')
                
        elif('note' in ans or 'notes' in ans):
            
            if('search' in ans):
                noteSearch()
            elif('record' or 'recording' in ans):
                noteTaking()
        elif('headlines' in ans or 'headline' in ans or 'news' in ans or 'dawn' in ans):
            readArticle()
                    
                
        print("done - result written to demo.wav")
        

# use the audio file as the audio source

#objectRecognition(imp,detector)
#textRecognition(imp)
#faceRecognition(imp)
        

