import cv2
import mediapipe as mp
import numpy as np

# Dictionary to store custom gestures and their corresponding contacts
custom_gestures = {}

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize MediaPipe drawing module to draw landmarks
mp_drawing = mp.solutions.drawing_utils

# Function to capture and return hand landmarks from the webcam
def get_landmarks(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks[0].landmark  # return the landmarks of the first detected hand
    return None

# Function to capture user's custom gesture mapping
def capture_custom_gesture():
    print("Capture a gesture and map it to a contact name:")
    contact_name = input("Enter contact name to map this gesture: ")
    print("Perform the gesture you want to assign to the contact.")
    return contact_name

# Function to compare current gesture with stored custom gestures
def compare_gesture(current_landmarks, stored_landmarks):
    # Here we use the relative distances between landmarks to compare gestures
    # This is a very simplified version of gesture comparison
    distance_threshold = 0.1  # Customize this threshold
    differences = []
    for i, lm in enumerate(current_landmarks):
        diff = np.abs(lm.x - stored_landmarks[i].x) + np.abs(lm.y - stored_landmarks[i].y) + np.abs(lm.z - stored_landmarks[i].z)
        differences.append(diff)
    total_diff = np.sum(differences)
    return total_diff < distance_threshold  # If the difference is below threshold, it's a match

# Initialize the webcam for capturing gestures
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
        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(frame, current_landmarks, mp_hands.HAND_CONNECTIONS)
        
        for gesture_name, stored_landmarks in custom_gestures.items():
            if compare_gesture(current_landmarks, stored_landmarks):
                # If the gesture matches, simulate the call to the mapped contact
                print(f"Recognized Gesture: {gesture_name}, Calling {custom_gestures[gesture_name]}")
                cv2.putText(frame, f"Calling {custom_gestures[gesture_name]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break  # Stop after the first match, you can allow multiple gesture recognition if needed
    
    # Display the frame with the hand landmarks and recognized gesture
    cv2.imshow("Hand Gesture Recognition", frame)
    
    # Press 'c' to capture a custom gesture, 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        contact_name = capture_custom_gesture()
        print("Perform your custom gesture now.")
        stored_landmarks = get_landmarks(frame)
        custom_gestures[contact_name] = stored_landmarks
        print(f"Custom gesture for {contact_name} saved!")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
