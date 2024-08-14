import tkinter as tk
from tkinter import filedialog, Toplevel, messagebox,OptionMenu,StringVar
from PIL import Image, ImageTk, ImageFont ,ImageDraw
import cv2
import face_recognition
import numpy as np
import pickle
import os
import threading
import datetime

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
        self.recognize_image_button = None  # New button for recognizing from image

        # Create the image object and store it as an instance variable
        self.load_menu = StringVar(self.window)
        self.load_menu.set("Load Encoding")  # Default text
        self.load_menu.trace("w", self.load_selected_encoding)

        encoding_files = self.get_encoding_files()
        self.load_menu_options = OptionMenu(self.window, self.load_menu, *encoding_files)
        self.load_menu_options.pack(pady=10)

    def create_widgets(self):
        background_image_path = "Images/background_image.jpg"
        self.background_image = Image.open(background_image_path)
        self.background_image = self.background_image.resize((self.screen_width, self.screen_height), Image.ANTIALIAS)
        self.background_image_tk = ImageTk.PhotoImage(self.background_image)

        self.background_label = tk.Label(self.window, image=self.background_image_tk)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.title_label = tk.Label(self.window, text="DIVYA NETRA", font=("Times New Roman", 39, "italic"),
                                    fg="white", bg="dark slate blue")
        self.title_label.pack(fill="x")

        # ... (existing code) ...

        self.load_menu = StringVar(self.window)
        self.load_menu.set("Load Encoding")  # Default text
        self.load_menu.trace("w", self.load_selected_encoding)

        encoding_files = self.get_encoding_files()
        self.load_menu_options = OptionMenu(self.window, self.load_menu, *encoding_files)
        self.load_menu_options.pack(pady=10)

    def get_encoding_files(self):
        # Get a list of all encoding files in the current directory
        encoding_files = [f for f in os.listdir() if f.startswith("encodings_") and f.endswith(".pickle")]
        return encoding_files

    def load_known_encodings_menu(self):
        # Load the default encoding file when the program starts
        encoding_files = self.get_encoding_files()
        if encoding_files:
            self.selected_encoding = encoding_files[0]
            self.load_known_encodings()

    def load_known_encodings(self):
        if self.selected_encoding:
            file_path = os.path.join(os.getcwd(), self.selected_encoding)
            if os.path.exists(file_path):
                self.data = pickle.loads(open(file_path, "rb").read())
                messagebox.showinfo("Success", f"Encoding file '{self.selected_encoding}' loaded successfully.")
            else:
                messagebox.showerror("Error", f"Selected encoding file '{self.selected_encoding}' not found.")

    def load_selected_encoding(self, *args):
        self.selected_encoding = self.load_menu.get()
        self.load_known_encodings()


    def create_widgets(self):
        background_image_path = "Images/background_image.jpg"
        self.background_image = Image.open(background_image_path)
        self.background_image = self.background_image.resize((self.screen_width, self.screen_height), Image.ANTIALIAS)
        self.background_image_tk = ImageTk.PhotoImage(self.background_image)  # Store the PhotoImage

        self.background_label = tk.Label(self.window, image=self.background_image_tk)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.title_label = tk.Label(self.window, text="DIVYA NETRA", font=("Times New Roman", 39, "italic"),
                                    fg="white", bg="dark slate blue")
        self.title_label.pack(fill="x")

        logo_image_path = "Images/NM.png"
        logo_image = Image.open(logo_image_path)
        logo_image = logo_image.resize((200, 54), Image.LANCZOS)
        self.logo_photo = ImageTk.PhotoImage(logo_image)

        logo_image_path = "Images/NM.png"
        logo_image = Image.open(logo_image_path)
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
        self.video_button = tk.Button(self.window, text="Real-time Recognizing", command=self.open_video_window,
                                      bg="dark slate blue", fg="white", font=("Times New Roman", 20),
                                      width=25, height=2)
        self.train_button.pack(pady=(0, 90))
        self.video_button.pack(pady=(50, 0))

        self.recognize_image_button = tk.Button(self.window, text="Recognize from Image", command=self.recognize_from_image,
                                                bg="dark slate blue", fg="white", font=("Times New Roman", 20),
                                                width=25, height=2)
        self.recognize_image_button.pack(pady=(50, 0))

        self.video_button = tk.Button(self.window, text="Recognize from Video", command=self.recognize_from_video,
                                  bg="dark slate blue", fg="white", font=("Times New Roman", 20),
                                  width=25, height=2)
        self.video_button.pack(pady=(50, 0))


    def load_known_encodings(self):
        if os.path.exists("encodings.pickle"):
            self.data = pickle.loads(open("encodings.pickle", "rb").read())


    def recognize_from_image(self):
        image_file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if image_file_path:
            image = face_recognition.load_image_file(image_file_path)

            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            if len(face_encodings) == 0:
                messagebox.showwarning("No Face Detected", "No face detected in the provided image.")
                return

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.data["encodings"], face_encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(self.data["encodings"], face_encoding)
                min_distance = min(face_distances)
                if min_distance <= self.threshold:
                    name = self.data["names"][np.argmin(face_distances)]
                else:
                    name = "Unknown"
                face_names.append(name)

            result_text = ""
            for name in face_names:
                result_text += name + "\n"

            if len(face_names) == 0:
                result_text = "No recognized face found in the provided image."

            # Call the display_recognized_image function to display the result
            self.display_recognized_image(image, face_encodings, face_names, result_text)
        else:
            messagebox.showwarning("No Image Selected", "Please select an image to recognize.")




  
    def display_recognized_image(self, image, face_encodings, face_names, result_text):
        window = Toplevel(self.window)
        window.title("Recognition Result")
        window.state('zoomed')

        # Get the dimensions of the result window
        result_width = window.winfo_screenwidth()
        result_height = window.winfo_screenheight()

        # Resize the image to fit the result window while maintaining the aspect ratio
        image_pil = Image.fromarray(image)
        image_pil.thumbnail((result_width, result_height))

        # Convert the image to RGB mode (required by Tkinter)
        image_pil_rgb = image_pil.convert("RGB")

        # Create a PhotoImage object from the resized image
        img_tk = ImageTk.PhotoImage(image=image_pil_rgb)

        # Calculate the positioning of the image to center it slightly higher in the window
        x_position = (result_width - image_pil.width) // 2
        y_position = (result_height - image_pil.height) // 3

        # Display the resized image in a label
        label = tk.Label(window, image=img_tk)
        label.image = img_tk
        label.place(x=x_position, y=y_position)

        # Create a new PhotoImage object with the updated image (with rectangles)
        img_with_rectangles = Image.fromarray(image)
        draw = ImageDraw.Draw(img_with_rectangles)

        # Draw rectangles around the recognized faces with the appropriate color
        for (top, right, bottom, left), name, face_encoding in zip(face_recognition.face_locations(image), face_names, face_encodings):
            face_distances = face_recognition.face_distance(self.data["encodings"], face_encoding)
            min_distance = min(face_distances)

            color = (0, 255, 0) if name != "Unknown" else (255, 0, 0)  # Green for recognized, Red for unknown
            draw.rectangle([left, top, right, bottom], outline=color, width=2)

            # Add text with name and confidence percentage
            confidence = int((1 - min_distance) * 100) if name != "Unknown" else 0
            text = f"{name} ({confidence}%)"
            text_width, text_height = draw.textsize(text, font=ImageFont.truetype("arial", 20))
            draw.rectangle([left, bottom, left + text_width, bottom + text_height], fill=color)
            draw.text((left, bottom), text, fill="black", font=ImageFont.truetype("arial", 20))
            img_tk_with_rectangles = ImageTk.PhotoImage(image=img_with_rectangles)

        # Update the label with the image including the rectangles and text
        label.config(image=img_tk_with_rectangles)
        label.image = img_tk_with_rectangles

        # Display the result text just below the image
        label_result = tk.Label(window, text=result_text, font=("Helvetica", 16),
                                fg="red" if "Unknown" in result_text else "blue")
        label_result.place(x=result_width // 2, y=y_position + image_pil.height + 20, anchor='center')
        

    def recognize_from_video(self):
        video_file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if not video_file_path:
            messagebox.showwarning("No Video Selected", "Please select a video file to recognize faces.")
            return

        self.status_label.config(text="Recognizing faces in the video...")

        def process_frames():
            ret, frame = self.video_capture.read()

            if not ret:
                # Video processing is complete
                self.status_label.config(text="Face recognition in the video completed.")
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

            self.video_label.config(image=img_tk)
            self.video_label.image = img_tk

            self.video_window.after(10, process_frames)

        self.video_window = Toplevel(self.window)
        self.video_window.title("Video Recognition")
        self.video_window.state('zoomed')

        self.video_label = tk.Label(self.video_window)
        self.video_label.pack()

        stop_button = tk.Button(self.video_window, text="Stop Video Recognition", command=self.stop_video_capture,
                                bg="#D70000", fg="white", font=("Helvetica", 14), width=25, height=2)
        stop_button.pack(pady=10)

        self.video_capture = cv2.VideoCapture(video_file_path)
        process_frames()


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
    
    

    def extract_face_region(self, image, face_landmarks):
        # Get the facial landmarks
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        top_lip = face_landmarks['top_lip']
        bottom_lip = face_landmarks['bottom_lip']

        # Calculate the bounding box for the face region
        left = min(left_eye[0][0], right_eye[0][0])
        top = min(left_eye[0][1], right_eye[0][1], top_lip[0][1], bottom_lip[0][1])
        right = max(left_eye[-1][0], right_eye[-1][0])
        bottom = max(left_eye[-1][1], right_eye[-1][1], top_lip[-1][1], bottom_lip[-1][1])

        # Crop the face region from the image
        face_image = image[top:bottom, left:right]

        return face_image

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
                    encodings_per_person = []

                    for file in os.listdir(image_folder):
                        full_file_path = os.path.join(image_folder, file)
                        image = cv2.imread(full_file_path)
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        boxes = face_recognition.face_locations(gray)
                        encodings = face_recognition.face_encodings(image, boxes)

                        for encoding in encodings:
                            encodings_per_person.append(encoding)
                            known_names.append(folder)

                    known_encodings.extend(encodings_per_person)

                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                file_name = f"encodings_{timestamp}.pickle"
                file_path = os.path.join(os.getcwd(), file_name)

                self.data = {"encodings": known_encodings, "names": known_names}

                with open(file_path, "wb") as f:
                    pickle.dump(self.data, f)

                self.status_label.config(text=f"Training completed. {count} faces trained.\nModel saved: {file_name}")
            else:
                self.status_label.config(text="Training canceled")

        # Start the training process in a separate thread
        training_thread = threading.Thread(target=training_thread)
        training_thread.start()

   


if __name__ == "__main__":
    app = DivyaNetra()
    app.window.mainloop()
