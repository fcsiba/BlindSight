from gtts import gTTS 
import cv2
import sys
import os
from playsound import playsound




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
         


    # Display the resulting frame
    #cv2.imshow('Video', frame)
    
    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break

# When everything is done, release the capture
#cv2.destroyAllWindows()