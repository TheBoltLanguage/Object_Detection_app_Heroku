import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
import tempfile

net = cv2.dnn.readNetFromDarknet('yolov3-tiny_obj.cfg', 'yolov3-tiny_obj_best.weights')
classes = ["Headphone", "Wallet"]
layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes),3))   
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()

st.title("Object Detection")
st.markdown("In this app, you can upload an image, video or open your webcam. And this app will detect wallet or headphone in your table.")
st.markdown("Below are the sample images which can be detected.The accuracy of the detected images is printed in the image, video or your webcam.")

st.image(["wallet.png","headphone1.png"], caption=["Sample picture of a Wallet","Sample picture of a Headphone"], width=300, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

choice = st.sidebar.radio("Choose an option",("Upload an image","Upload a video", "Choose your webcam"))
st.subheader("Output is seen here below.")
if choice == "Upload an image":
    frame1 = st.sidebar.file_uploader("Upload",type=['jpg','png','jpeg'])
    if frame1:
        frame = Image.open(frame1)
        frame = np.array(frame)
        height, width, channels = frame.shape
        # detecting objects
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
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6,.4)
        a=""

        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x,y), (x+w, y+h), color,2)
                a=a+" "+label
                #a = str(round(confidence,2))
                cv2.putText(frame, label + ' ' + str(round(confidence,2)), (x,y+30), font,1, color,2)
            
        st.image(frame)
        st.write("Detected object: "+a)

if choice == "Upload a video":
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    frame = st.sidebar.file_uploader("Upload",type=["mp4"])
    frame_id = 0
    font = cv2.FONT_HERSHEY_PLAIN
    no_of_classes = []
    frame_window = st.image([])
    frame_text = st.markdown("")
    frame_text1 = st.markdown("")
    frame_text2 = st.markdown("")
    frame_text3 = st.markdown("")
    if frame:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(frame.read())
        vf = cv2.VideoCapture(tfile.name)
        while vf:
            _,frame=vf.read()
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
            elapsed_time = time.time() - starting_time
            fps=frame_id/elapsed_time
            img_np = np.array(frame)
            frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            frame1 = cv2.putText(frame, 'FPS:'+str(round(fps,2)), (10,50), font, 2, (0,0,0),1)
            frame_window.image(frame)
            

if choice == "Choose your webcam":
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    run=st.sidebar.checkbox('Open/Close your Webcam')
    video = cv2.VideoCapture(-1)
    frame_id = 0
    frame_window = st.image([])
    frame_text = st.markdown("")
    frame_text1 = st.markdown("")
    frame_text2 = st.empty()
    frame_text3 = st.markdown("")
    frame_text4 = st.empty()
    while run:
        _,frame=video.read()

        height, width, channels = frame.shape
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
        t=""
        t_ = ""
        n=0
        n_ = 0
        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x,y), (x+w, y+h), color,2)
                a = str(round(confidence,2))
                cv2.putText(frame, label + ' ' + str(round(confidence,2)), (x,y+30), font, 1, (255,255,255),2)
                t = t + " " + label
                n+= 1    
            else:
                label_=str(classes[~class_ids[i]])
                t_ = t_ + " " + label_
                       
        elapsed_time = time.time() - starting_time
        fps=frame_id/elapsed_time
        img_np = np.array(frame)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        frame1 = cv2.putText(frame, 'FPS:'+str(round(fps,2)), (10,50), font, 2, (0,0,0),1)
        frame_window.image(frame1)
        frame_text3.text("Items Found : "+t)
        frame_text4.text("No. of Items Found: "+str(n))
