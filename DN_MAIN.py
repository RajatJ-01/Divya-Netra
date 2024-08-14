import tkinter as tk
import cv2
import os
from PIL import ImageTk, Image
import subprocess
import time
import threading
from DN_FACE import DivyaNetra



class CriminalFaceRecognitionApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("DIVYA NETRA")
        self.window.state('zoomed')
       

        # Load the background image
        background_image_path = "Images/background_image.jpg"
        background_image = Image.open(background_image_path)
        background_image = background_image.resize((self.window.winfo_screenwidth(), self.window.winfo_screenheight()), Image.LANCZOS)
        self.background_photo = ImageTk.PhotoImage(background_image)

        # Load the logo image
        logo_image_path = "Images/NM.png"
        logo_image = Image.open(logo_image_path)
        logo_image = logo_image.resize((200, 53), Image.LANCZOS)
        self.logo_photo = ImageTk.PhotoImage(logo_image)

        logo_image_path = "Images/NM.png"
        logo_image = Image.open(logo_image_path)
        logo_image = logo_image.resize((200, 53), Image.LANCZOS)
        self.logo_photo = ImageTk.PhotoImage(logo_image)

        # Create a Canvas widget with the background image
        self.canvas = tk.Canvas(self.window, width=self.window.winfo_screenwidth(), height=self.window.winfo_screenheight())
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, image=self.background_photo, anchor="nw")

        # Create a label with the specified text, font size, and font family
        label = tk.Label(self.canvas, text="DIVYA NETRA", font=("Times New Roman", 35), bg="dark slate blue", fg="white", width=self.window.winfo_screenwidth())
        label.place(relx=0.5, rely=0.04, anchor="center")

        # Create a label for the logo
        logo_label = tk.Label(self.canvas, image=self.logo_photo, bg="dark slate blue")
        logo_label.place(relx=0.0, rely=0.0)



        logo_label = tk.Label(self.canvas, image=self.logo_photo, bg="dark slate blue")
        logo_label.place(relx=1.0, rely=0.0, anchor="ne")

        # Create the Register button at the upper middle position
        register_button = tk.Button(self.canvas, text="Register Face", font=("Times New Roman", 30), bg="dark slate blue", fg="white", command=self.open_register_window)
        register_button.place(relx=0.5, rely=0.2, anchor="center")

        # Create the Start Recognizing button below the Register button
        start_recognizing_button = tk.Button(self.canvas, text="Start Recognizing", font=("Times New Roman", 30), bg="dark slate blue", fg="white", command=self.start_recognizing)
        start_recognizing_button.place(relx=0.5, rely=0.35, anchor="center")

        self.captured_label = None  # Initialize captured_label as None

        # Initialize a flag to indicate whether to continue capturing images
        self.continue_capturing = True

    def open_register_window(self):
        register_window = tk.Toplevel(self.window)
        register_window.title("Register")

        # Get the screen dimensions
        screen_width = register_window.winfo_screenwidth()
        screen_height = register_window.winfo_screenheight()

        # Set the registration window size to full screen size
        register_window.geometry(f"{screen_width}x{screen_height}")

        # Create label and entry for name
        name_label = tk.Label(register_window, text="Name:")
        name_label.pack()
        self.name_entry = tk.Entry(register_window)
        self.name_entry.pack()

        # Create label and entry for age
        age_label = tk.Label(register_window, text="Age:")
        age_label.pack()
        self.age_entry = tk.Entry(register_window)
        self.age_entry.pack()

        # Create a button to capture face
        capture_button = tk.Button(register_window, text="Capture Face", font=("Times New Roman", 16), command=self.capture_face)
        capture_button.pack()

        # Create a label to display "Captured" message on the capturing window
        self.captured_label = tk.Label(register_window, text="", font=("Times New Roman", 20), fg="red", bg="black")
        self.captured_label.place(relx=0.5, rely=0.9, anchor="center")

        # Create the Stop Capturing button
        stop_button = tk.Button(register_window, text="Stop Capturing", font=("Times New Roman", 16), command=self.stop_capture)
        stop_button.pack()

    def stop_capture(self):
        # Stop capturing images when the stop button is clicked
        self.continue_capturing = False

    def capture_face(self):
        # ... (existing code remains unchanged)
        name = self.name_entry.get()
        age = self.age_entry.get()

        # Create a directory for the dataset if it doesn't exist
        dataset_folder = "dataset"
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)

        # Create a directory with the provided name
        user_folder = os.path.join(dataset_folder, name)
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)

        # Create a video capture object
        capture = cv2.VideoCapture(0)

        # Initialize a counter for capturing images
        image_counter = 0

        # Function to capture images after a 3-second interval
        def capture_images():
            ret, frame = capture.read()

            # Display the resulting frame
            cv2.imshow("Capturing", frame)

            # Construct the file name
            file_name = f"{user_folder}/{name}_{age}_{self.image_counter}.jpg"

            # Save the image
            cv2.imwrite(file_name, frame)

            # Increment the image counter
            self.image_counter += 1

            # Display "Captured" on the capturing window
            self.captured_label.config(text="Captured")
            self.window.update()  # Update the label text immediately

            # After 1 second, reset the label to an empty string
            self.window.after(1000, lambda: self.captured_label.config(text=""))

            print(f"Image {file_name} saved.")

            # If 20 images are captured or stop button is pressed, stop the camera and show the message
            if self.image_counter >= 20 or not self.continue_capturing:
                print("Pictures have been successfully captured.")
                capture.release()
                cv2.destroyAllWindows()
            else:
                # Schedule the next image capture after 3 seconds
                self.window.after(3000, capture_images)

        # Increment the image counter
        self.image_counter = 0

        # Start capturing images after 3 seconds
        self.window.after(3000, capture_images)

    def start_recognizing(self):
      #result = subprocess.run(['python', 	'divya_netra_app.py'],capture_output=True, text=True)
    #   def recognize_thread():
    #         result = subprocess.run(['python', 'face.py'],capture_output=True, text=True)
    #         print(result.stdout)

    #     # Start the recognition thread
    #   recognizing_thread = threading.Thread(target=recognize_thread)
    #   recognizing_thread.start() 
        self.window.destroy()
        DivyaNetra()
       


    def run(self):
        self.window.mainloop()

# Create an instance of the CriminalFaceRecognitionApp and run the application
app = CriminalFaceRecognitionApp()
app.run()
