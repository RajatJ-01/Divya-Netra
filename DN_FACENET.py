import tkinter as tk
from tkinter import filedialog, Toplevel, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import pickle
import os
import threading
import queue
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

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

        # Load MTCNN and ArcFace models
        self.mtcnn = MTCNN(keep_all=True)
        self.face_rec_model = InceptionResnetV1(pretrained='vggface2').eval()

        self.create_widgets()
        self.load_known_encodings()

    def create_widgets(self):
        background_image_path = "Images/background_image.jpg"
        self.background_image = Image.open(background_image_path)
        # Updated to use Image.Resampling.LANCZOS instead of Image.ANTIALIAS
        self.background_image = self.background_image.resize(
            (self.screen_width, self.screen_height), Image.Resampling.LANCZOS
        )
        self.background_image_tk = ImageTk.PhotoImage(self.background_image)

        self.background_label = tk.Label(self.window, image=self.background_image_tk)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.title_label = tk.Label(self.window, text="DIVYA NETRA", font=("Times New Roman", 39, "italic"),
                                    fg="white", bg="dark slate blue")
        self.title_label.pack(fill="x")

        logo_image_path = "Images/NM.png"
        logo_image = Image.open(logo_image_path)
        logo_image = logo_image.resize((200, 54), Image.Resampling.LANCZOS)
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
        self.video_button = tk.Button(self.window, text="Real-time Recognizing", command=self.open_video_window,
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
        faces = self.mtcnn.detect(rgb_frame)[0]
        
        if faces is None:
            return [], []
        
        encodings = []
        for face in faces:
            face_img = rgb_frame[int(face[1]):int(face[3]), int(face[0]):int(face[2])]
            if face_img.size == 0:
                continue
            face_img = cv2.resize(face_img, (160, 160))
            face_img = (face_img / 255.0).astype(np.float32)
            face_img = np.expand_dims(face_img, axis=0)
            face_embedding = self.face_rec_model(torch.tensor(face_img).permute(0, 3, 1, 2)).detach().numpy()
            encodings.append(face_embedding.flatten())
        
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
                distances = [np.linalg.norm(known_encoding - face_encoding) for known_encoding in self.data["encodings"]]
                min_distance = min(distances) if distances else float('inf')
                if min_distance <= self.threshold:
                    name = self.data["names"][np.argmin(distances)]
                else:
                    name = "Unknown"
                face_names_batch.append(name)

            for (face, name) in zip(face_locations_batch, face_names_batch):
                (left, top, right, bottom) = (face[0], face[1], face[2], face[3])
                color = (0, 255, 0) if name != "Unknown" else (255, 0, 0)
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), color, 2)
                cv2.rectangle(frame, (int(left), int(bottom) - 35), (int(right), int(bottom)), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                confidence = int((1 - min_distance) * 100) if name != "Unknown" else 0
                text = f"{name} ({confidence}%)"
                cv2.putText(frame, text, (int(left) + 6, int(bottom) - 6), font, 0.8, (0, 0, 0), 1)

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

        path = filedialog.askdirectory()
        if path:
            self.train_button.config(state=tk.DISABLED)  # Disable the Train Model button during training

            def training_thread():
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
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        faces = self.mtcnn.detect(rgb_image)[0]

                        for face in faces:
                            face_img = rgb_image[int(face[1]):int(face[3]), int(face[0]):int(face[2])]
                            if face_img.size == 0:
                                continue
                            face_img = cv2.resize(face_img, (160, 160))
                            face_img = (face_img / 255.0).astype(np.float32)
                            face_img = np.expand_dims(face_img, axis=0)
                            face_embedding = self.face_rec_model(torch.tensor(face_img).permute(0, 3, 1, 2)).detach().numpy()
                            known_encodings.append(face_embedding.flatten())
                            known_names.append(folder)

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
            self.status_label.config(text="Training completed. {} faces trained".format(count))
            self.train_button.config(state=tk.NORMAL)  # Re-enable the Train Model button after training
        except queue.Empty:
            self.window.after(100, self.check_queue)

    def main(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = DivyaNetra()
    app.main()
