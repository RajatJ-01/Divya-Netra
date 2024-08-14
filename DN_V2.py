import tkinter as tk
from tkinter import filedialog, Toplevel, messagebox
from PIL import Image, ImageTk
import cv2
import dlib
import numpy as np
import pickle
import os
import threading
import queue

class DivyaNetra:
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
        self.queue = queue.Queue()
        
        # Load dlib models
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
        
        self.create_widgets()
        self.load_known_encodings()

    def create_widgets(self):
        background_image_path = "Images/background_image.jpg"
        self.background_image = Image.open(background_image_path)
        # Replace Image.ANTIALIAS with Image.LANCZOS
        self.background_image = self.background_image.resize((self.screen_width, self.screen_height), Image.LANCZOS)
        self.background_image_tk = ImageTk.PhotoImage(self.background_image)

        self.background_label = tk.Label(self.window, image=self.background_image_tk)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.title_label = tk.Label(self.window, text="DIVYA NETRA", font=("Times New Roman", 39, "italic"),
                                    fg="white", bg="dark slate blue")
        self.title_label.pack(fill="x")

        logo_image_path = "Images/NM.png"
        logo_image = Image.open(logo_image_path)
        logo_image = logo_image.resize((200, 54), Image.LANCZOS)
        self.logo_photo = ImageTk.PhotoImage(logo_image)

        logo_label = tk.Label(self.window, image=self.logo_photo, bg="dark slate blue")
        logo_label.place(relx=1.0, rely=0.0, anchor="ne")
        logo_label = tk.Label(self.window, image=self.logo_photo, bg="dark slate blue")
        logo_label.place(relx=0.0, rely=0.0)

        self.status_label = tk.Label(self.window, text="Start Training", fg="green", font=("Helvetica", 14))
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
        if self.video_capture:
            self.video_capture.release()
        if self.video_window:
            self.video_window.destroy()

    def encode_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector(rgb_frame)
        encodings = []
        
        for face in faces:
            shape = self.predictor(rgb_frame, face)
            face_descriptor = self.face_rec_model.compute_face_descriptor(rgb_frame, shape)
            encodings.append(np.array(face_descriptor))
        
        return faces, encodings

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
                matches = [np.linalg.norm(known_encoding - face_encoding) for known_encoding in self.data["encodings"]]
                min_distance = min(matches) if matches else float('inf')
                if min_distance <= self.threshold:
                    name = self.data["names"][np.argmin(matches)]
                else:
                    name = "Unknown"
                face_names_batch.append(name)

            for (face, name) in zip(face_locations_batch, face_names_batch):
                (left, top, right, bottom) = (face.left(), face.top(), face.right(), face.bottom())
                color = (0, 255, 0) if name != "Unknown" else (255, 0, 0)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                confidence = int((1 - min_distance) * 100) if name != "Unknown" else 0
                text = f"{name} ({confidence}%)"
                cv2.putText(frame, text, (left + 6, bottom - 6), font, 0.8, (0, 0, 0), 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)

            label.config(image=img_tk)
            label.image = img_tk

            self.video_window.after(10, process_frames)

        process_frames()

    def train_faces(self):
        self.status_label.config(text="Training in progress...")
        self.window.update_idletasks()

        path = filedialog.askdirectory()
        if path:
            self.train_button.config(state=tk.DISABLED) 

            def training_thread():
                known_encodings = []
                known_names = []
                count = 0

                for folder in os.listdir(path):
                    image_folder = os.path.join(path, folder)

                    if not os.path.isdir(image_folder):
                        continue

                    count += 1
                    encodings_per_person = []

                    for file in os.listdir(image_folder):
                        full_file_path = os.path.join(image_folder, file)
                        image = cv2.imread(full_file_path)
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        faces = self.detector(rgb_image)

                        for face in faces:
                            shape = self.predictor(rgb_image, face)
                            encoding = self.face_rec_model.compute_face_descriptor(rgb_image, shape)
                            encodings_per_person.append(encoding)
                            known_names.append(folder)

                    known_encodings.extend(encodings_per_person)

                self.data = {"encodings": known_encodings, "names": known_names}

                with open("encodings.pickle", "wb") as f:
                    pickle.dump(self.data, f)

                self.queue.put(count)

            self.queue = queue.Queue()
            training_thread = threading.Thread(target=training_thread)
            training_thread.start()
            self.window.after(100, self.check_queue)
        else:
            self.status_label.config(text="Training canceled")

    def check_queue(self):
        try:
            count = self.queue.get_nowait()
            self.status_label.config(text="Training completed")
            self.train_button.config(state=tk.NORMAL)  
        except queue.Empty:
            self.window.after(100, self.check_queue)

    def main(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = DivyaNetra()
    app.main()
