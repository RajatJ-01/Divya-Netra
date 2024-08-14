import cv2

def capture_and_extract_frames(video_file, output_folder, frame_interval=1):
    # Create a VideoCapture object to open the video file
    cap = cv2.VideoCapture(video_file)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set the initial frame number to 0
    frame_number = 0

    # Loop through the video frames with a given interval
    while frame_number < total_frames:
        # Read the next frame from the video
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            break

        # Save the frame as an image in the output folder
        output_file = f"{output_folder}/frame_{frame_number:04d}.png"
        cv2.imwrite(output_file, frame)

        # Print the progress
        print(f"Extracting frame {frame_number+1}/{total_frames}")

        # Increment the frame number by the interval
        frame_number += frame_interval

        # Set the next frame position to read
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Release the VideoCapture object and close the video file
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file_path = "Videos/manish.mp4"
    output_folder_path = "Frames"
    frame_interval = 10  # Adjust this value to control the frame extraction interval
    
    capture_and_extract_frames(video_file_path, output_folder_path, frame_interval)
