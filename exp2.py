import dlib
import cv2
import numpy as np

def align_face(image_path):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found or could not be loaded.")
        return None

    # Initialize dlib's face detector and landmark predictor
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Detect faces in the image
    faces = face_detector(image, 1)

    if len(faces) == 0:
        print("No face detected in the image.")
        return None

    # Get the facial landmarks for the first detected face
    landmarks = landmark_predictor(image, faces[0])
    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

    # Calculate the center of the eyes
    left_eye_center = np.mean(landmarks[36:42], axis=0)
    right_eye_center = np.mean(landmarks[42:48], axis=0)

    # Calculate the angle between the eyes (yaw)
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # Rotate the image to align the face
    rotation_matrix = cv2.getRotationMatrix2D(tuple(left_eye_center), angle, 1.0)
    aligned_face = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]),
                                  flags=cv2.INTER_CUBIC)

    return aligned_face

def create_side_facing_images(aligned_face, num_images, max_rotation=45):
    # Generate side-facing images by rotating the aligned face
    side_facing_images = []
    for i in range(num_images):
        # Randomly select a rotation angle within the specified range
        angle = np.random.randint(-max_rotation, max_rotation)

        # Rotate the aligned face by the selected angle
        rotated_face = rotate_face(aligned_face, angle)

        # Append the rotated face to the list of side-facing images
        side_facing_images.append(rotated_face)

    return side_facing_images

def rotate_face(face, angle):
    height, width, _ = face.shape
    center = (width // 2, height // 2)

    # Rotate the face around its center
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_face = cv2.warpAffine(face, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC)

    return rotated_face

if __name__ == "__main__":
    front_facing_image_path = "front_facing_image.jpg"

    # Step 1: Face Alignment
    aligned_face = align_face(front_facing_image_path)

    if aligned_face is not None:
        # Step 4: Create side-facing images
        num_side_facing_images = 5  # Number of side-facing images to create
        side_facing_images = create_side_facing_images(aligned_face, num_side_facing_images)

        # Display the original aligned face
        cv2.imshow("Aligned Face", aligned_face)

        # Display the side-facing images
        for i, side_facing_image in enumerate(side_facing_images):
            cv2.imshow(f"Side Facing Image {i+1}", side_facing_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
