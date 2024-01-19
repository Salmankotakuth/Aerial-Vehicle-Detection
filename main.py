#
# import cv2
# from ultralytics import YOLO
#
# # Load the YOLOv8 model
# model = YOLO('best.pt')
#
# # Open the video file
# video_path = "Static Drone.mp4"
# cap = cv2.VideoCapture(video_path)
#
# # Get video properties
# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# # Define the codec and create a VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec according to your needs
# output_path = "C:/Users/salma/Desktop/areal/output.avi"
# out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#
# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()
#
#     if success:
#         # Run YOLOv8 inference on the frame
#         results = model.track(frame, show=True,  hide_labels=True, conf=0.3)
#
#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()
#
#         # Save the annotated frame to the output video
#         out.write(annotated_frame)
#
#         # Display the annotated frame
#         # cv2.imshow("YOLOv8 Inference", annotated_frame)
#
#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break
#
# # Release the video capture object, the VideoWriter object, and close the display window
# cap.release()
# out.release()
# cv2.destroyAllWindows()


import cv2
from ultralytics import YOLO


model = YOLO("best.pt")


cap = cv2.VideoCapture("Drone Footage.mp4")

out = cv2.VideoWriter('VisDrone.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (int(cap.get(3)), int(cap.get(4))))


while True:
    ret, frame = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.predict(frame)  # only detect car, truck and bus
    boxes = results[0].boxes.xyxy.cpu()
    clss = results[0].boxes.cls.cpu().tolist()

    for box, cls in zip(boxes, clss):
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # calculate center
        cx = int(box[0] + box[2]) // 2
        cy = int(box[1] + box[3]) // 2

        # Plot center
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    out.write(frame)

    resized_frame = cv2.resize(frame, (1020, 550))

    cv2.imshow("vehicle distance calculation", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()