import tkinter as tk
from tkinter import filedialog, Toplevel
from PIL import Image, ImageTk
import cv2
import face_recognition
import numpy as np
import pickle
import os
import threading

class DivyaNetraApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("DIVYA NETRA")
        self.window.state('zoomed')
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()
        self.threshold = 0.5
        self.frame_count = 0
        self.skip_frames = 4
        self.data = None
        self.video_capture = None
        self.video_window = None
        self.background_image = None
        self.background_label = None
        self.background_image_tk = None
        self.title_label = None
        self.status_label = None
        self.train_button = None
        self.video_button = None
        self.back_button = None

        # Create the image object and store it as an instance variable
       
        self.create_widgets()
        self.load_known_encodings()

    def create_widgets(self):
        self.background_image = Image.open("background_image.jpg")
        self.background_image = self.background_image.resize((self.screen_width, self.screen_height), Image.ANTIALIAS)
        self.background_image_tk = ImageTk.PhotoImage(self.background_image)  # Store the PhotoImage

        self.background_label = tk.Label(self.window, image=self.background_image_tk)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.title_label = tk.Label(self.window, text="DIVYA NETRA", font=("Times New Roman", 39, "italic"),
                                    fg="white", bg="dark slate blue")
        self.title_label.pack(fill="x")

        logo_image = Image.open("NM.png")
        logo_image = logo_image.resize((200, 54), Image.LANCZOS)
        self.logo_photo = ImageTk.PhotoImage(logo_image)

        logo_image = Image.open("NM.png")
        logo_image = logo_image.resize((200, 54), Image.LANCZOS)
        self.logo_photo = ImageTk.PhotoImage(logo_image)

        logo_label = tk.Label(self.window, image=self.logo_photo, bg="dark slate blue")
        logo_label.place(relx=1.0, rely=0.0, anchor="ne") 

        logo_label = tk.Label(self.window, image=self.logo_photo, bg="dark slate blue")
        logo_label.place(relx=0.0, rely=0.0)


        self.status_label = tk.Label(self.window, text="Ready", fg="green", font=("Helvetica", 14))
        self.status_label.pack(pady=10, side="top")

        self.train_button = tk.Button(self.window, text="Train Model", command=self.train_faces,
                                      bg="dark slate blue", fg="white", font=("Times New Roman", 20),
                                      width=25, height=2)
        self.video_button = tk.Button(self.window, text="Start Recognizing", command=self.open_video_window,
                                      bg="dark slate blue", fg="white", font=("Times New Roman", 20),
                                      width=25, height=2)
        self.train_button.pack(pady=(0, 90))
        self.video_button.pack(pady=(50, 0))


    def load_known_encodings(self):
        if os.path.exists("encodings.pickle"):
            self.data = pickle.loads(open("encodings.pickle", "rb").read())

    def open_video_window(self):
        self.video_window = Toplevel(self.window)
        self.video_window.title("Video Capture")
        self.video_window.state('zoomed')

        video_label = tk.Label(self.video_window)
        video_label.pack()

        stop_button = tk.Button(self.video_window, text="Stop Video Capture", command=self.stop_video_capture,
                                bg="#D70000", fg="white", font=("Helvetica", 14), width=25, height=2)
        stop_button.pack(pady=10)

        self.start_video_capture(video_label)

    def stop_video_capture(self):
        self.video_capture.release()
        self.video_window.destroy()

    def encode_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        return face_locations, face_encodings

    def start_video_capture(self, label):
        self.video_capture = cv2.VideoCapture(0)

        def process_frames():
            ret, frame = self.video_capture.read()

            self.frame_count += 1

            if self.frame_count % self.skip_frames != 0:
                self.video_window.after(10, process_frames)
                return

            face_locations_batch, face_encodings_batch = self.encode_faces(frame)

            face_names_batch = []
            for face_encoding in face_encodings_batch:
                matches = face_recognition.compare_faces(self.data["encodings"], face_encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(self.data["encodings"], face_encoding)
                min_distance = min(face_distances)
                if min_distance <= self.threshold:
                    name = self.data["names"][np.argmin(face_distances)]
                else:
                    name = "Unknown"
                face_names_batch.append(name)

            for (top, right, bottom, left), face_names in zip(face_locations_batch, face_names_batch):
                top = top * 4
                right = right * 4
                bottom = bottom * 4
                left = left * 4
                color = (0, 255, 0) if face_names != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX

                confidence = int((1 - min_distance) * 100) if face_names != "Unknown" else 0
                text = f"{face_names} ({confidence}%)"
                cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.8, (0, 0, 0), 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)

            label.config(image=img_tk)
            label.image = img_tk

            self.video_window.after(10, process_frames)

        process_frames()

    def train_faces(self):
        self.status_label.config(text="Please wait for some time...")
        self.window.update_idletasks()

        def training_thread():
            path = filedialog.askdirectory()
            if path:
                known_encodings = []
                known_names = []
                count = 0

                for folder in os.listdir(path):
                    image_folder = os.path.join(path, folder)

                    if not os.path.isdir(image_folder):
                        continue

                    count += 1
                    for file in os.listdir(image_folder):
                        full_file_path = os.path.join(image_folder, file)
                        image = cv2.imread(full_file_path)
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        boxes = face_recognition.face_locations(gray)
                        encodings = face_recognition.face_encodings(image, boxes)

                        for encoding in encodings:
                            known_encodings.append(encoding)
                            known_names.append(folder)

                self.data = {"encodings": known_encodings, "names": known_names}

                with open("encodings.pickle", "wb") as f:
                    pickle.dump(self.data, f)

                self.status_label.config(text="Training completed. {} faces trained".format(count))
            else:
                self.status_label.config(text="Training canceled")

        # Start the training process in a separate thread
        training_thread = threading.Thread(target=training_thread)
        training_thread.start()
   


if __name__ == "__main__":
    app = DivyaNetraApp()
    app.window.mainloop()
