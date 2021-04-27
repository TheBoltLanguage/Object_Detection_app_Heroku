################################################## Importing Library ##############################################################
import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
import tempfile
################################################# Required library imported #######################################################
######################################## Creating function for anchorbox, class, and class_ids #######################################
def fun():
    height,width,channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0,0,0), swapRB =True, crop = False)
    net.setInput(blob)
    outs = net.forward(outputlayers)
    boxes = []
    confidences = []
    class_ids = []
    for output in outs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > .6:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold,.4)
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x,y), (x+w, y+h), color,2)
            a = str(round(confidence,2))
            cv2.putText(frame, label + ' ' + str(round(confidence,2)), (x,y+30), font, 1, (255,255,255),2)
    return class_ids
#########################################################################################################################################
st.title("Object Detection")
st.markdown("In this app, you can upload an image, video or open your webcam (The webcam works only when you run the app in the local machine. Get the app from github repo https://github.com/haaruhito/Object_Detection_app_Streamlit). And this app will detect two objects that is wallet or headphone. The accuracy is relatively ok however more training data is required.For better accuracy, you can use the images similar to sample images given below.")
st.markdown("Below are the sample images which can be detected.The accuracy of the detected images is printed in the image, video or your webcam.")
st.image(["wallet.png","headphone1.png"], caption=["Sample picture of a Wallet","Sample picture of a Headphone"], width=300, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
st.subheader("Output is seen here below.")
st.sidebar.markdown("# Model")
confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
net = cv2.dnn.readNetFromDarknet('yolov3-tiny_obj.cfg', 'yolov3-tiny_obj_best.weights')
classes = ["Headphone", "Wallet"]
layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes),3))
#########################################################################################################################################
choice = st.sidebar.radio("Choose an option",("Upload an image","Upload a video", "Choose your webcam"))
starting_time = time.time()
frame_id = 0
font = cv2.FONT_HERSHEY_PLAIN
no_of_classes = []
frame_window = st.image([])
frame_text = st.markdown("")
frame_text1 = st.markdown("")
frame_text2 = st.markdown("")
frame_text3 = st.markdown("")
####################################### Uploading the image ############################################################################
if choice == "Upload an image":
    frame = st.sidebar.file_uploader("Upload",type=["jpeg", "jpg", "png", "jfif"])
    if frame:
        frame1 = Image.open(frame)
        frame = np.array(frame1)
        if frame1:
            class_ids = fun()
            elapsed_time = time.time() - starting_time
            fps=frame_id/elapsed_time
            img_np = frame
            frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            frame1 = cv2.putText(frame, 'FPS:'+str(round(fps,2)), (10,50), font, 2, (0,0,0),1)
            frame_window.image(frame1)
            class_name = ([classes[x] for x in class_ids])
            len_cl = len(class_name)
            no_of_classes.append(len_cl)
            separator = ", "
            frame_text2.markdown('Items: ' + separator.join(map(str, class_name)))
            frame_text3.markdown('Number of items: '+str(len_cl))
#######################################################################################################################################
##################################################### Uploading Video ##################################################################
if choice == "Upload a video":
    frame = st.sidebar.file_uploader("Upload",type=["mp4","mov","wmv"])
    if frame:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(frame.read())
        vf = cv2.VideoCapture(tfile.name)
        while vf:
            _,frame=vf.read()
            class_ids = fun()
            elapsed_time = time.time() - starting_time
            fps=frame_id/elapsed_time
            #code to change the blue intensive video to normal
            img_np = np.array(frame)
            frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            #adding the frame to above created empty frame_window
            frame1 = cv2.putText(frame, 'FPS:'+str(round(fps,2)), (10,50), font, 2, (0,0,0),1)
            frame_window.image(frame1)
            class_name = ([classes[x] for x in class_ids])
            len_cl = len(class_name)
            no_of_classes.append(len_cl)
            separator = ", "
            frame_text2.markdown('Items: ' + separator.join(map(str, class_name)))
            frame_text3.markdown('Number of items: '+str(len_cl))
#####################################################################################################################################
############################################## Using webcam for the object detection ################################################
if choice == "Choose your webcam":
    run=st.sidebar.checkbox('Open/Close your Webcam')
    video = cv2.VideoCapture(0)
    while run:
        _,frame=video.read()
        class_ids = fun()
        elapsed_time = time.time() - starting_time
        fps=frame_id/elapsed_time
        #code to change the blue intensive video to normal
        img_np = np.array(frame)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        #adding the frame to above created empty frame_window
        frame1 = cv2.putText(frame, 'FPS:'+str(round(fps,2)), (10,50), font, 2, (0,0,0),1)
        frame_window.image(frame1)
        class_name = ([classes[x] for x in class_ids])
        len_cl = len(class_name)
        no_of_classes.append(len_cl)
        separator = ", "
        frame_text2.markdown('Items: ' + separator.join(map(str, class_name)))
        frame_text3.markdown('Number of items: '+str(len_cl))
####################################################################################################################################

       