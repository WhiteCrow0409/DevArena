import cv2
import mediapipe as mp
from pose_detection.pose_estimator import process_frame, Exercise
import threading
import time

mp_pose = mp.solutions.pose
cap = None
cap_lock = threading.Lock()

exercise_state = {
    'type': None,
    'exercise': Exercise(),
    'pose': None,
    'estimator': None
}

def reset_exercise_state():
    """Reset the exercise state to initial values."""
    exercise_state['exercise'] = Exercise()
    exercise_state['estimator'] = None

def initialize_camera():
    """Initialize the camera connection."""
    global cap
    with cap_lock:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

def set_exercise(exercise_type, duration):
    """Set the exercise type and duration."""
    reset_exercise_state()  # Reset the state first
    exercise_state['type'] = exercise_type
    exercise_state['exercise'] = Exercise()
    # Set start time to now and calculate end time
    current_time = time.time()
    exercise_state['exercise'].start_time = current_time + duration
    # Initialize MediaPipe pose if not already
    if not exercise_state.get('pose'):
        exercise_state['pose'] = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    # Create new PoseEstimator instance
    exercise_state['estimator'] = None  # Will be created on first frame

def get_results():
    """Get the current exercise results."""
    exercise = exercise_state['exercise']
    estimator = exercise_state.get('estimator')
    
    # If we have an estimator, use its values
    if estimator:
        exercise.counter = estimator.counter
        exercise.stage = estimator.stage
        exercise.accuracy_score = estimator.accuracy_score
        exercise.form_feedback = estimator.form_feedback
    
    # Calculate remaining time
    if exercise.start_time:
        remaining = max(0, exercise.start_time - time.time())
        exercise.time_remaining = remaining
        
        # Reset exercise if time is up
        if remaining <= 0:
            reset_exercise_state()
    else:
        remaining = 0
    
    return {
        'reps': exercise.counter,
        'stage': exercise.stage or 'Ready',
        'accuracy': exercise.accuracy_score,
        'total_accuracy': exercise.total_accuracy,
        'feedback': exercise.form_feedback or 'Get ready to start',
        'time_remaining': remaining,
        'type': exercise_state['type']
    }

def get_exercise_state():
    """Get current exercise state."""
    return exercise_state

def generate_frames():
    """Capture and process video frames for pose detection."""
    initialize_camera()
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            with cap_lock:
                if cap is None or not cap.isOpened():
                    break
                    
                success, frame = cap.read()
                if not success:
                    continue

                # Process frame for pose detection if exercise type is set
                if exercise_state['type']:
                    frame, counter, stage, accuracy, total_accuracy = process_frame(
                        frame,
                        pose,
                        exercise_state['type'],
                        exercise_state['exercise']
                    )
                    
                    # Draw landmarks and connections
                    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.pose_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                            mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2)
                        )

                try:
                    # Use higher quality JPEG encoding
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                    _, buffer = cv2.imencode('.jpg', frame, encode_param)
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(f"Error encoding frame: {e}")
                    continue

def cleanup():
    """Release video capture resources."""
    global cap
    with cap_lock:
        if cap is not None:
            cap.release()
            cap = None
