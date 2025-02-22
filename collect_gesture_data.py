import os
import cv2
import numpy as np
import mediapipe as mp
import time
import os
  # Suppresses all TensorFlow logs

  # Suppress warnings and info logs from TensorFlow

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Directory to save data
gesture_data_dir = "gesture_data"
os.makedirs(gesture_data_dir, exist_ok=True)

def sanitize_input(input_str):
    """
    Sanitize user input to remove invalid characters for folder names.
    """
    return ''.join(c if c.isalnum() or c == '_' else '_' for c in input_str)

def collect_data(contact_name, gesture_name, time_limit=10):
    """
    Collects gesture data for a specific contact and gesture name.
    Saves hand landmarks as .npy files in a dedicated directory.

    Args:
        contact_name (str): The name of the contact associated with the gesture.
        gesture_name (str): The gesture name (e.g., "Thumbs_Up").
        time_limit (int): Maximum time (in seconds) to capture landmarks.
    """
    print(f"Starting data collection for contact: {contact_name}, gesture: {gesture_name}...")

    # Sanitize and prepare directory path
    contact_name = sanitize_input(contact_name)
    gesture_name = sanitize_input(gesture_name)
    gesture_dir = os.path.normpath(os.path.join(gesture_data_dir, f"{contact_name}_{gesture_name}"))
    os.makedirs(gesture_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam. Please check your device.")
        cv2.destroyAllWindows()
        return
    else:
        print("Webcam initialized successfully.")

    sample_count = 0
    start_time = time.time()

    while cap.isOpened():
        # Check for time limit
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:
            print("Time limit reached. Stopping data collection.")
            break

        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from webcam.")
            break

        # Flip and process frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract and save landmarks
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                np.save(os.path.join(gesture_dir, f"{sample_count}.npy"), landmarks)
                sample_count += 1

                # Draw hand landmarks on the frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the frame
        cv2.putText(frame, f"Collecting {gesture_name}: Sample {sample_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Time left: {int(time_limit - elapsed_time)}s", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Gesture Data Collection", frame)

        # Exit if 'q' is pressed or 50 samples are collected
        if cv2.waitKey(1) & 0xFF == ord('q') or sample_count >= 50:
            print("Data collection completed.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {sample_count} samples for {contact_name} - {gesture_name} in {gesture_dir}")

# Main program
if __name__ == "__main__":
    # Prompt the user for contact name and gesture name

    print("Please enter the contact name and gesture name.")
    contact_name = input("Enter contact name: ").strip()  # Ask once
    print("Please enter the contact name and gesture name.")
    contact_name = input("Enter contact name: ").strip()  # Ask once
    gesture_name = input("Enter gesture name (e.g., Thumbs_Up, Peace_Sign): ").strip()  # Ask once

    collect_data(contact_name, gesture_name, time_limit=10)

