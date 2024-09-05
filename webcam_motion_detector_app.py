import cv2
import time
import glob
import os
import smtplib
#import imghdr
from email.message import EmailMessage
from threading import Thread
import streamlit as st
from datetime import datetime

# Motion Detection
#video_path = "C:\\Users\\SIDDHARTH RAJKUMAR B\\Desktop\\CVIP Project dataset\\Videos Captured\\3.mp4"
#video = cv2.VideoCapture(video_path)
video = cv2.VideoCapture(0)
time.sleep(1)
first_frame = None
status_list = []
count = 1
image_with_object = None

def clean_folder():
    print("clean_folder function started")
    images = glob.glob("images/*.png")
    for image in images:
        os.remove(image)
    print("clean_folder function ended")

clean_thread = Thread(target=clean_folder)
clean_thread.daemon = True
clean_thread_started = False

# Email Sending
SENDER = ""
RECEIVER = ""

def send_email(image_path, sender_email, sender_password, receiver_email):
    if image_path is None:
        print("Error: image_path is None")
        return

    email_message = EmailMessage()
    email_message["Subject"] = "New customer showed up!"
    email_message.set_content("Hey, we just saw a new customer!")

    with open(image_path, "rb") as file:
        content = file.read()

    # Convert the image to RGB before attaching to the email
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    _, buffer = cv2.imencode(".png", image_rgb)
    image_content = buffer.tobytes()
    email_message.add_attachment(image_content, maintype="image", subtype="png")

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as gmail:
            gmail.starttls()
            gmail.login(sender_email, sender_password)
            print("Login successful")
            gmail.sendmail(sender_email, receiver_email, email_message.as_string())
            print("Email sent successfully")

    except Exception as e:
        print(f"Error sending email: {e}")

# Streamlit Webcam App
st.title("Motion Detector")

# Get user input for email and password
sender_email = st.text_input("Enter your email address:")
sender_password = st.text_input("Enter your email password:", type="password")
receiver_email = st.text_input("Enter recipient's email address:")

start = st.button('Start Camera')

if start and sender_email and sender_password and receiver_email:
    streamlit_image = st.image([])
    camera = video

    while start:
        check, frame = camera.read()

        # Check if the frame is empty
        if not check:
            print("Error: Empty frame")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now = datetime.now()
        cv2.putText(img=frame, text=now.strftime("%A"), org=(30, 80), fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=3, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(img=frame, text=now.strftime("%H:%M:%S"), org=(30, 140), fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=3, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        streamlit_image.image(frame)

        status = 0
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_gau = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        if first_frame is None:
            first_frame = gray_frame_gau
        delta_frame = cv2.absdiff(first_frame, gray_frame_gau)
        thresh_frame = cv2.threshold(delta_frame, 60, 255, cv2.THRESH_BINARY)[1]
        dil_frame = cv2.dilate(thresh_frame, None, iterations=2)

        cv2.imshow("My video", dil_frame)

        contours, _ = cv2.findContours(dil_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 5000:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            rectangle = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            if rectangle.any():
                status = 1
                cv2.imwrite(f"images/{count}.png", frame)
                count = count + 1
                all_images = glob.glob("images/*.png")
                if all_images:
                    index = min(len(all_images) // 2, len(all_images) - 1)
                    image_with_object = all_images[index]

        status_list.append(status)
        status_list = status_list[-2:]

        if status_list[0] == 1 and status_list[1] == 0 and image_with_object is not None:
            email_thread = Thread(target=send_email, args=(image_with_object, sender_email, sender_password, receiver_email))
            email_thread.daemon = True
            email_thread.start()

            if not clean_thread_started:
                clean_thread.start()
                clean_thread_started = True

            image_with_object = None

        print(status_list)

        cv2.imshow("Video", frame)
        key = cv2.waitKey(1)

        if key == ord("q"):
            start = False

    camera.release()
    cv2.destroyAllWindows()
