from ultralytics import YOLO
from PIL import Image
import cv2
import supervision as sv
import smtplib
import pandas as pd
import os
from time import time
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from Email_settings import password, from_email, to_email, to_number
from datetime import datetime




def send_email(to_email, from_email, alert):
    message = MIMEMultipart()
    message['From'] = from_email
    message['To'] = to_email
    message['Subject'] = "System Alert"
    message_body = f'{alert}'
    message.attach(MIMEText(message_body, 'plain'))
    server_ssl.sendmail(from_email, to_email, message.as_string())



def send_text_msg(to_number, from_email, alert):
    #https://en.wikipedia.org/wiki/SMS_gateway
    text_message: MIMEMultipart = MIMEMultipart()
    text_message['Subject'] = "Hey"
    text_message_body = f'{alert}'
    text_message.attach(MIMEText(text_message_body, 'plain'))
    server_ssl.sendmail(from_email, to_number, text_message.as_string())


def main():
    #User variables to control alerts and output
    alerts_flag_on = False
    #user input on which class objects to alert on
    # [] = None ; 0 = Aircraft ; 1 = Birds ; 2 = Drone ; 3 = Helicopter ; 4 = Jet ; 5 = Contrail
    objects_class_to_alert_on = []
    #root folder name
    root_folder_name=f'./Results-GX100015-exiftool'
    # output text filename, can be anything
    output_txt_filename = 'Detections.csv'
    # output video name, can be anything
    video_output_filename = f'Detection_Video.avi'
    # output video fps
    output_fps = 20
    model =YOLO("yolov8n.pt")
    #model = YOLO("yolov8n.pt")
    #model = YOLO("LPRv1.pt")

    # user input minimum confidence threshold for predictions
    model_confidence = 0.55
    # Webcam
    source = './Video-clips/GX100015.mp4'
    # source = 'https://www.youtube.com/live/N9P24SBeu4E'
    # source = "./Contrail-Example-5.mp4"
    # Single stream with batch-size 1 inference
    #source = 'rtsp://192.168.5.34/axis-media/media.amp'  # RTSP, RTMP, TCP or IP streaming address
    # Multiple streams with batched inference (i.e. batch-size 8 for 8 streams)
    # source = 'path/to/list.streams'  # *.streams text file with one streaming address per row



    # intialize variables for logic
    previous_frame_time = 0
    alert_sent = False

    #results main folder path
    main_results_folder =f'{root_folder_name}-{datetime.utcnow().strftime("%B_%d_%Y")}'
    # if main folder for results does not exist then create it
    if not os.path.exists(f'{main_results_folder}'):
        os.makedirs(f'{main_results_folder}')


    #Capture video from source, read source fps
    cap =cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    font =cv2.FONT_HERSHEY_SIMPLEX

    #defines capture resolution size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    #define output video file codec and variable
    fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D") #mp4v
    out = cv2.VideoWriter(f'{main_results_folder}/{video_output_filename}', fourcc, output_fps, (1280, 720))

    ######################################################
    # Email_settings contains the from and to configuration
    # Below is the method to log in
    ######################################################
    if alerts_flag_on:
        server_ssl = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server_ssl.ehlo()
        server_ssl.login(from_email, password)


    #create output text file, print column titles
    with open(f'{main_results_folder}/{output_txt_filename}', "a") as f:
        print('Alarm (YES/NO)', 'Time(UTC)', 'Track_ID', 'Class_Name', 'Confidence', file=f)

        while cap.isOpened():
            day_time_now = datetime.utcnow().strftime('%B %d %Y - %H:%M:%S')
            img_filename = datetime.utcnow().strftime('%B_%d_%Y_%H_%M_%S.jpg')

            ############################################################
            #read video frame, report true or false for variable success
            ############################################################
            success, frame = cap.read()

            ###################################################
            #if video or source stream is unavailable, exit code
            ###################################################
            if not success:
                exit(0)

            ##########################
            #calculate fps from source
            ##########################
            new_frame_time = time()
            fps = 1/(new_frame_time-previous_frame_time)
            fps= int(fps)
            previous_frame_time=new_frame_time


            #save fps as string to print to screen
            fps_indication = f'FPS: {fps}'



            ##################################################################################################
            #run prediction for trained model on current frame, show visual window, save detections to results
            #
            #stream_buffer  --  manages computer memory more effciently and should be used when streaming
            #persist        --  facilitates tracks between frames, needed when using model.track vs model.predict
            #conf           --  sets the minimum threshold for object class detections
            ##################################################################################################
            results = model.track(frame, stream_buffer=False, persist=False, classes=0, conf=model_confidence)
 

            #save annotated frames to video file
            annotated_frame = results[0].plot()
            cv2.putText(annotated_frame, fps_indication, (20,70),font, 1.0, (100,255, 0), 2)
            cv2.putText(annotated_frame, str(day_time_now), (260,70),font, 0.8, (100,255, 0), 2)
            annotated_frame = cv2.resize(annotated_frame, (1280, 720))

            #cv2.imshow("Object Detection Window", annotated_frame)






            ###########################################################
            #Save results detected from frame, save to row in csv file
            ###########################################################
            for result in results:
                boxes = result.boxes.cpu().numpy()

                #loop through all the boxes detected in a frame, save and print info
                for box in range(len(boxes.cls)):
                    class_number = int(boxes.cls[box])
                    class_name = result.names[class_number]
                    confidence = int(boxes.conf[box]*100)
                    bx = boxes.xyxy[box].tolist()
                    alert = f'Detected a {class_name} on {day_time_now} UTC'
                    #if a track established on object persisting between frames, save track id, record video frames
                    if boxes.is_track:
                        track_id = int(boxes.id[box])
                        out.write(annotated_frame)
                    else:
                        track_id = f'No Tracks'

                    if alerts_flag_on:
                        # If a class ID occurs we want to alert on and track is established and
                        # alert not yet sent, send alert, set alert_sent flag
                        # Once object or track is lost, reset alert_sent flag
                        if class_number in objects_class_to_alert_on and boxes.is_track:
                            if not alert_sent:
                                send_email(to_email, from_email, alert)
                                send_text_msg(to_number, from_email, alert)
                                alert_sent = True
                                print(F'ALERT-ALERT, {day_time_now},{track_id},{class_name},{confidence}, Detected a {class_name}', file=f)





                    #print detections into comma delimited format to a text file
                    print(F'-----------, {day_time_now},{track_id},{class_name},{confidence}', file=f)

                    #set the path for cropped image to reside
                    path = f'{main_results_folder}/{class_name}/'

                    #if cropped image folder does not exist then create it
                    if not os.path.exists(path):
                        os.makedirs(path)


                    # capture and save the cropped images
                    im_array = result.plot(labels=True, boxes=bx)
                    im = Image.fromarray(im_array[..., ::-1])
                    #im = Image.fromarray(annotated_frame)
                    im.save(path+img_filename, "JPEG")



            #When there are no longer tracks between frames, reset alert flag
            if not boxes.is_track:
                alert_sent = False

            ##########################################################
            #if Esc key pressed for at least 30 milliseconds, exit code
            ##########################################################
            if (cv2.waitKey(30) == 27):
                break
                #close the email stmp server
        #server_ssl.quit()
        cv2.destroyAllWindows()
        out.release()
        cap.release()
    f.close()



if __name__== "__main__":
    main()















