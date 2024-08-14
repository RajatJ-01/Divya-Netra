import tkinter as tk
from tkinter import filedialog, Toplevel, messagebox, ttk
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
        
        # Load dlib models with error handling
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            self.face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
            self.window.quit()
        
        self.create_widgets()
        self.load_known_encodings()

    def create_widgets(self):
        background_image_path = "Images/background_image.jpg"
        if os.path.exists(background_image_path):
            self.background_image = Image.open(background_image_path)
            self.background_image = self.background_image.resize((self.screen_width, self.screen_height), Image.LANCZOS)
            self.background_image_tk = ImageTk.PhotoImage(self.background_image)
            self.background_label = tk.Label(self.window, image=self.background_image_tk)
            self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.title_label = tk.Label(self.window, text="DIVYA NETRA", font=("Times New Roman", 39, "italic"),
                                    fg="white", bg="dark slate blue")
        self.title_label.pack(fill="x")

        logo_image_path = "Images/NM.png"
        if os.path.exists(logo_image_path):
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

        self.progress_bar = ttk.Progressbar(self.window, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(pady=(10, 0))

    def load_known_encodings(self):
        if os.path.exists("encodings.pickle"):
            try:
                self.data = pickle.loads(open("encodings.pickle", "rb").read())
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load known encodings: {str(e)}")
                self.data = {"encodings": [], "names": []}

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

    def align_face(self, img, shape):
        eye_left = (shape.part(36).x, shape.part(36).y)
        eye_right = (shape.part(45).x, shape.part(45).y)
        dY = eye_right[1] - eye_left[1]
        dX = eye_right[0] - eye_left[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        desired_left_eye = (0.35, 0.35)
        desired_face_width = 256
        desired_face_height = 256
        desired_right_eye_x = 1.0 - desired_left_eye[0]

        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desired_dist = (desired_right_eye_x - desired_left_eye[0]) * desired_face_width
        scale = desired_dist / dist

        eyes_center = ((eye_left[0] + eye_right[0]) // 2,
                       (eye_left[1] + eye_right[1]) // 2)

        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        tX = desired_face_width * 0.5
        tY = desired_face_height * desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        (w, h) = (desired_face_width, desired_face_height)
        aligned_face = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
        return aligned_face

    def augment_image(self, image):
        flipped = cv2.flip(image, 1)
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
        return [image, flipped, bright]

    def encode_faces(self, image):
        faces = []
        encodings = []

        detected_faces = self.detector(image, 1)
        for face in detected_faces:
            shape = self.predictor(image, face)
            aligned_face = self.align_face(image, shape)
            face_descriptor = self.face_rec_model.compute_face_descriptor(aligned_face, shape)
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

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.configure(image=imgtk)
            label.update()

            self.video_window.after(10, process_frames)

        process_frames()

    def save_encodings(self, data):
        with open("encodings.pickle", "wb") as f:
            pickle.dump(data, f)

    def validate_image_file(self, file_path):
        valid_extensions = ['.jpg', '.jpeg', '.png']
        if not os.path.exists(file_path) or not any(file_path.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Invalid file: {file_path}")

    def train_faces(self):
        directory_path = filedialog.askdirectory(title="Select Folder Containing Face Images")
        if not directory_path:
            return

        self.progress_bar["value"] = 0
        self.progress_bar["maximum"] = len(os.listdir(directory_path))
        self.progress_bar.pack()

        def training_process():
            names = []
            encodings = []

            for idx, folder_name in enumerate(os.listdir(directory_path)):
                folder_path = os.path.join(directory_path, folder_name)

                if not os.path.isdir(folder_path):
                    continue

                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    try:
                        self.validate_image_file(file_path)
                        image = cv2.imread(file_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        for augmented_image in self.augment_image(image):
                            faces, encoding = self.encode_faces(augmented_image)
                            if encoding:
                                encodings.append(encoding[0])
                                names.append(folder_name)

                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to process file {file_path}: {str(e)}")

                self.progress_bar["value"] = idx + 1
                self.window.update_idletasks()

            if encodings and names:
                self.data["encodings"].extend(encodings)
                self.data["names"].extend(names)
                self.save_encodings(self.data)
                messagebox.showinfo("Success", "Training completed successfully!")
            else:
                messagebox.showwarning("Warning", "No faces were detected during training.")

            self.progress_bar.pack_forget()

        # Run the training process in a separate thread
        threading.Thread(target=training_process).start()

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = DivyaNetra()
    app.run()
