import os
import cv2
import numpy as np
import mediapipe as mp
import time

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Directory to save gesture data
gesture_data_dir = "gesture_data"
os.makedirs(gesture_data_dir, exist_ok=True)

# Dictionary to store custom gestures and their corresponding contacts
custom_gestures = {}

# Function to sanitize input for directory names
def sanitize_input(input_str):
    return ''.join(c if c.isalnum() or c == '_' else '_' for c in input_str)

# Function to capture and return hand landmarks from the webcam
def get_landmarks(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks[0].landmark  # return the landmarks of the first detected hand
    return None

# Function to compare current gesture with stored custom gestures
def compare_gesture(current_landmarks, stored_landmarks):
    distance_threshold = 0.1  # Customize this threshold
    differences = []
    for i, lm in enumerate(current_landmarks):
        diff = np.abs(lm.x - stored_landmarks[i].x) + np.abs(lm.y - stored_landmarks[i].y) + np.abs(lm.z - stored_landmarks[i].z)
        differences.append(diff)
    total_diff = np.sum(differences)
    return total_diff < distance_threshold

# Function to collect gesture data
def collect_data(contact_name, gesture_name, time_limit=10):
    print(f"Starting data collection for contact: {contact_name}, gesture: {gesture_name}...")
    
    contact_name = sanitize_input(contact_name)
    gesture_name = sanitize_input(gesture_name)
    gesture_dir = os.path.normpath(os.path.join(gesture_data_dir, f"{contact_name}_{gesture_name}"))
    os.makedirs(gesture_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam. Please check your device.")
        cv2.destroyAllWindows()
        return

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

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                np.save(os.path.join(gesture_dir, f"{sample_count}.npy"), landmarks)
                sample_count += 1

                # Draw landmarks on the frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

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

# Function to capture user's custom gesture mapping
def capture_custom_gesture():
    print("Capture a gesture and map it to a contact name:")
    contact_name = input("Enter contact name to map this gesture: ")
    print("Perform the gesture you want to assign to the contact.")
    return contact_name

# Main program
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Get landmarks for the current frame
        current_landmarks = get_landmarks(frame)

        # If landmarks are detected, check against stored custom gestures
        if current_landmarks:
            mp_draw.draw_landmarks(frame, current_landmarks, mp_hands.HAND_CONNECTIONS)

            for gesture_name, stored_landmarks in custom_gestures.items():
                if compare_gesture(current_landmarks, stored_landmarks):
                    print(f"Recognized Gesture: {gesture_name}, Calling {custom_gestures[gesture_name]}")
                    cv2.putText(frame, f"Calling {custom_gestures[gesture_name]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    break

        # Display the frame with the hand landmarks and recognized gesture
        cv2.imshow("Hand Gesture Recognition", frame)

        # Press 'c' to capture a custom gesture, 'd' to collect gesture data, 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            contact_name = capture_custom_gesture()
            print("Perform your custom gesture now.")
            stored_landmarks = get_landmarks(frame)
            custom_gestures[contact_name] = stored_landmarks
            print(f"Custom gesture for {contact_name} saved!")
        elif key == ord('d'):
            print("Collecting gesture data...")
            contact_name = input("Enter contact name: ").strip()
            gesture_name = input("Enter gesture name (e.g., Thumbs_Up): ").strip()
            collect_data(contact_name, gesture_name, time_limit=10)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
