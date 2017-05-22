# Proof-of-concept
import cv2
import sys
from constants import *
from emotion_recognition import EmotionRecognition
import numpy as np
import glob
import Tkinter, tkFileDialog
from Tkinter import *
ch = 0
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

def brighten(data,b):
     datab = data * b
     return datab    

def format_image(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  faces = cascade_classifier.detectMultiScale(
      image,
      scaleFactor = 1.3,
      minNeighbors = 5
  )
  # None is we don't found an image
  if not len(faces) > 0:
    return None
  max_area_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
      max_area_face = face
  # Chop image to face
  face = max_area_face
  image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
  # Resize image to network size
  try:
    image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation = cv2.INTER_CUBIC) / 255.
  except Exception:
    print("[+] Problem during resize")
    return None
  # cv2.imshow("Lol", image)
  # cv2.waitKey(0)
  return image

# Load Model
network = EmotionRecognition()
network.build_network()
feelings_faces = []
for index, emotion in enumerate(EMOTIONS):
  feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))

def sel():
    
    if(var.get() == 1):
                  selection = "You selected the web cam"
                  label.config(text = "You selected the web cam",font=('times', 20, 'bold'))
                  root = Tkinter.Tk()
                  print "your web cam is gettig start....."
		  print "_________________________________"
		  video_capture = cv2.VideoCapture(0)
		  font = cv2.FONT_HERSHEY_SIMPLEX


		    #print emotion

		  while True:
		    # Capture frame-by-frame	
		    ret, frame = video_capture.read()

		    # Predict result with network
		    result = network.predict(format_image(frame))
		    if result is not None:
		      a = result
		      for i in range (len(a[0])):
			      print (EMOTIONS[i] +" = "+ str(round((float(a[0][i]*100)),2))+' %') 

		      print ("Final emotion on your face is ....."+EMOTIONS[result[0].index(max(result[0]))])
		      print ("**************************################***********************************")

		    #print (130 + int(result[0][index] * 100)
		    # Draw face in frame
		    # for (x,y,w,h) in faces:
		    #   cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

		    # Write results in frame
		    if result is not None:
		      a = result
		      for index, emotion in enumerate(EMOTIONS):
		        cv2.putText(frame, (emotion +" ="+str(round((float(a[0][index]*100)),2))+"%"), (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 1, (102, 102, 0), 1);
		        cv2.rectangle(frame, (160, index * 20 + 10), (160 + int(result[0][index] * 100), (index + 1) * 20 + 4), (0, 0, 102), -1)
		       # print ( index * 20 + 20)
		      face_image = feelings_faces[result[0].index(max(result[0]))]
		      cv2.putText(frame,EMOTIONS[result[0].index(max(result[0]))], (10,346), cv2.FONT_HERSHEY_PLAIN, 3.0,(0,255,128) , thickness=4)
		      print EMOTIONS[result[0].index(max(result[0]))]
		      
		      # Ugly transparent fix
		      for c in range(0, 3):
		        frame[200:320, 10:130, c] = face_image[:,:,c] * (face_image[:, :, 3] / 255.0) +  frame[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)


		    # Display the resulting frame
		    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
		    cv2.imshow("output", frame)
              
		    if cv2.waitKey(1) & 0xFF == ord('q'):
		      break

		  # When everything is done, release the capture
		  video_capture.release()
		  cv2.destroyAllWindows()
    else:
                  selection = "You selected the Image folder"
                  label.config(text = "You selected the Image folder",font=('times', 20, 'bold'))
                  root = Tkinter.Tk()
		  dirname = tkFileDialog.askdirectory(parent=root,initialdir="/",title='Please select a directory')
		  if len(dirname ) > 0:
		     print "You chose %s" % dirname 
		     print "___________________________________________________________________________________________________________"
		  for img in glob.glob(dirname+("/*.*")):
		        frame = cv2.imread(img)


		#Predict result with network
			
			result = network.predict(format_image(frame))
			if result is not None:
		             a = result
		             for i in range (len(a[0])):
			         print (EMOTIONS[i] +" = "+ str(round((float(a[0][i]*100)),2))+' %') 
		             print ("Final emotion on your face is ....."+EMOTIONS[result[0].index(max(result[0]))])
		             #print ("**************************################***********************************")

			
			if result is not None:
			    a = result
			    for index, emotion in enumerate(EMOTIONS):
			      cv2.putText(frame, (emotion +" ="+str(round((float(a[0][index]*100)),2))+"%"), (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 1, (102, 102, 0), 1);
			      cv2.rectangle(frame, (160, index * 20 + 10), (160 + int(result[0][index] * 100), (index + 1) * 20 + 4), (0, 0, 102), -1)
			     # print ( index * 20 + 20)
			    face_image = feelings_faces[result[0].index(max(result[0]))]
                            height, width = frame.shape[:2]
			    cv2.putText(frame,EMOTIONS[result[0].index(max(result[0]))], (30,(height/8)+130), cv2.FONT_HERSHEY_PLAIN, 3.0,(0,255,128) , thickness=4)
			    print EMOTIONS[result[0].index(max(result[0]))]
			    
			    # Ugly transparent fix
			    #for c in range(0, 3):
			     #frame[200:320, 10:130, c] = face_image[:,:,c] * (face_image[:, :, 3] / 255.0) +  frame[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)


			  # Display the resulting frame

			cv2.namedWindow('image', cv2.WINDOW_NORMAL)
			cv2.imshow('output',frame)
			cv2.waitKey(0)
		        cv2.destroyAllWindows()

                  

  
root = Tk()


var = IntVar()
logo1 = PhotoImage(file = "/home/raj/final_project/emotion-recognition-neural-networks-master/ui_images/c.png")
logo2 = PhotoImage(file = "/home/raj/final_project/emotion-recognition-neural-networks-master/ui_images/f.png")
tex1 = Label(root, 
      text="""Choose your choice:""",
      justify = LEFT,pady =20,
      padx = 100)
tex1.config(bg='lightgreen', font=('times', 40, 'bold'))
tex1.pack()

Radiobutton(root, text="web cam",image = logo1, variable=var, value=1, command=sel).pack(anchor=W,pady=20,padx = 40)
Radiobutton(root, text="image folder",image = logo2, variable=var, value=2, command=sel).pack(anchor=W,pady=20,padx = 40)

label = Label(root)
label.pack()
def close_window(): 
    root.destroy()

frame = Frame(root)
frame.pack()
button = Button (frame, text = "QUIT", command = close_window)
button.pack(side=TOP, expand=YES)
root.mainloop()
