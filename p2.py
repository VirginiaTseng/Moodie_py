import cv2
from ultralytics import YOLO
import time
import pyttsx3

import asyncio

speaking=False

IP_URL="./p1.mp4"
# Load the YOLO model
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(IP_URL)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Known width of the object (e.g., a car width in meters)
KNOWN_WIDTH = 1.8  # Example width in meters

# Focal length of the camera (calibrated)
FOCAL_LENGTH = 800  # Example focal length in pixels

async def texttospeech(strs):
    engine = pyttsx3.init()
    engine.say(strs)
    engine.runAndWait()

def_speak_gap=20
speak_gap=def_speak_gap

while cap.isOpened():
    success, im0 = cap.read()
    if speaking or not success:
        print("Video frame is empty or video processing has been successfully completed.")
        speak_gap-=1
        print("skipping a frame")
        if(speak_gap<0):
            speaking=False
            continue
        else:
            continue
        break
    results = model(im0)
    for r in results:
        for box in r.boxes:
            cls = box.cls
            conf = box.conf
            if conf >= 0.5:
                print(box.xyxy)
                # Calculate the width of the bounding box in pixels
                box_width = box.xyxy[0][1] - box.xyxy[0][0]
                # Calculate the distance
                distance = (KNOWN_WIDTH * FOCAL_LENGTH) / box_width
                print(f"Object: {model.names[int(cls)]}, Distance: {distance:.2f} meters")
                if distance<5:
                    asyncio.run(texttospeech(model.names[int(cls)]+" is too close"))
                    speaking=True
                    speak_gap=def_speak_gap
    
# # Load an image
# image_path = r"015.jpg"
# image = cv2.imread(image_path)

# # Perform inference
# results = model(image)


# # Iterate through the results and calculate distances
# for r in results:
#     for box in r.boxes:
#         cls = box.cls
#         conf = box.conf
#         if conf >= 0.5:
#             print(box.xyxy)
#             # Calculate the width of the bounding box in pixels
#             box_width = box.xyxy[0][1] - box.xyxy[0][0]
#             # Calculate the distance
#             distance = (KNOWN_WIDTH * FOCAL_LENGTH) / box_width
#             print(f"Object: {model.names[int(cls)]}, Distance: {distance:.2f} meters")
#             if distance< 5:
#                 texttospeech(model.names[int(cls)]+" is too close")
#                 time.sleep(2)

# # Display the image with bounding boxes and distances
# for r in results:
#     r.plot()
# cv2.imshow("Image", image)
# time.sleep(2)
# #cv2.waitKey(0)
# cv2.destroyAllWindows()