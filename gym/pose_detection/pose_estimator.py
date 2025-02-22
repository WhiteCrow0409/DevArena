import cv2
import mediapipe as mp
import numpy as np
import time

class Exercise:
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.form_feedback = ""
        self.accuracy_score = 0
        self.perfect_reps = 0
        self.start_time = None
        self.time_remaining = 0
        self.total_accuracy = 0

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Exercise state variables
        self.counter = 0
        self.stage = None
        self.exercise_type = None
        self.form_feedback = ""
        self.accuracy_score = 0
        self.perfect_reps = 0
        
        # Angle thresholds
        self.BICEP_CURL_DOWN = 160
        self.BICEP_CURL_UP = 70
        self.SQUAT_STANDING = 170
        self.SQUAT_DOWN = 110

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                 np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle

    def detect_pose(self, frame, exercise_type="bicep_curl"):
        """Detect pose and count reps for specified exercise"""
        self.exercise_type = exercise_type
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return frame, {"counter": self.counter, "stage": "No pose detected", "feedback": "No pose detected"}
        
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Draw pose landmarks
        self.mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=self.mp_draw.DrawingSpec(color=(245,66,230), thickness=2)
        )
        
        # Process specific exercise
        if exercise_type == "bicep_curl":
            return self._process_bicep_curl(frame, landmarks)
        elif exercise_type == "squat":
            return self._process_squat(frame, landmarks)
        
        return frame, {"counter": 0, "stage": "Invalid exercise", "feedback": "Invalid exercise type"}

    def _process_bicep_curl(self, frame, landmarks):
        """Process bicep curl exercise"""
        # Get coordinates for right arm
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                   landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        # Calculate angle
        angle = self.calculate_angle(shoulder, elbow, wrist)
        
        # Draw angle on frame
        cv2.putText(frame, f"Angle: {int(angle)}",
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        # Calculate accuracy based on form
        accuracy = 100.0
        
        # Check elbow position stability
        hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        elbow_hip_distance = abs(elbow[0] - hip.x)
        if elbow_hip_distance > 0.1:  # If elbow moves too far from body
            accuracy -= 10
            self.form_feedback = "Keep your elbow close to your body"
        
        # Check shoulder stability
        if abs(shoulder[1] - self.last_shoulder_y if hasattr(self, 'last_shoulder_y') else shoulder[1]) > 0.02:
            accuracy -= 5
            self.form_feedback = "Keep your shoulder steady"
        self.last_shoulder_y = shoulder[1]
        
        # Count reps and update accuracy
        if angle > self.BICEP_CURL_DOWN:
            self.stage = "down"
            self.form_feedback = "Lower the weight slowly"
        elif angle < self.BICEP_CURL_UP and self.stage == 'down':
            self.stage = "up"
            self.counter += 1
            
            # Perfect rep if form maintained
            if accuracy >= 96:
                self.perfect_reps += 1
                self.form_feedback = "Perfect rep! Keep it up"
            else:
                self.form_feedback = "Good rep! Watch your form"
            
            # Update overall accuracy
            self.accuracy_score = (self.perfect_reps / self.counter) * 100 if self.counter > 0 else 0
        
        return frame, {
            "counter": self.counter,
            "stage": self.stage,
            "feedback": self.form_feedback,
            "accuracy": round(self.accuracy_score, 2)
        }

    def _process_squat(self, frame, landmarks):
        """Process squat exercise"""
        # Get coordinates for right leg and spine
        hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
               landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                   landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        
        # Calculate angle
        angle = self.calculate_angle(hip, knee, ankle)
        
        # Draw angle on frame
        cv2.putText(frame, f"Angle: {int(angle)}",
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        # Calculate accuracy based on form
        accuracy = 100.0
        
        # Check knee alignment
        knee_x = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x
        ankle_x = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
        if abs(knee_x - ankle_x) > 0.1:  # Threshold for knee alignment
            accuracy -= 15
            self.form_feedback = "Keep your knees aligned with your toes"
        
        # Check back angle
        spine_angle = self.calculate_angle(
            [shoulder[0], shoulder[1]],
            [hip[0], hip[1]],
            [hip[0], hip[1] - 0.5]  # Virtual point above hip for vertical reference
        )
        if abs(90 - spine_angle) > 20:  # More than 20 degrees tilt
            accuracy -= 10
            self.form_feedback = "Keep your back straight"
        
        # Count reps and update accuracy
        if angle > self.SQUAT_STANDING:
            self.stage = "standing"
            self.form_feedback = "Maintain good posture"
        elif angle < self.SQUAT_DOWN and self.stage == 'standing':
            self.stage = "down"
            self.counter += 1
            
            # Perfect rep if form maintained
            if accuracy >= 96:
                self.perfect_reps += 1
                self.form_feedback = "Perfect squat! Great form"
            else:
                self.form_feedback = "Good depth! Watch your form"
            
            # Update overall accuracy
            self.accuracy_score = (self.perfect_reps / self.counter) * 100 if self.counter > 0 else 0
        
        return frame, {
            "counter": self.counter,
            "stage": self.stage,
            "feedback": self.form_feedback,
            "accuracy": round(self.accuracy_score, 2)
        }

    def reset_counter(self):
        """Reset the rep counter"""
        self.counter = 0
        self.stage = None
        self.form_feedback = ""

# Test function
def test_pose_estimator():
    cap = cv2.VideoCapture(0)
    estimator = PoseEstimator()
    exercise_type = "bicep_curl"  # or "bicep_curl"
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        frame, data = estimator.detect_pose(frame, exercise_type)
        
        # Draw information on frame
        cv2.putText(frame, f'Reps: {data["counter"]}', 
                    (15,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f'Stage: {data["stage"]}', 
                    (15,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f'Feedback: {data["feedback"]}', 
                    (15,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv2.imshow('Exercise Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def process_frame(frame, pose, exercise_type, exercise_state):
    """Process a video frame for pose detection and exercise tracking."""
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    if not results.pose_landmarks:
        return frame, exercise_state.counter, "No pose detected", 0, exercise_state.total_accuracy
    
    # Get the landmarks for visualization
    landmarks = results.pose_landmarks.landmark

    # Create a PoseEstimator instance if not exists
    if not hasattr(exercise_state, 'estimator'):
        exercise_state.estimator = PoseEstimator()
        exercise_state.estimator.counter = exercise_state.counter
        exercise_state.estimator.stage = exercise_state.stage
        exercise_state.estimator.perfect_reps = exercise_state.perfect_reps
        exercise_state.estimator.accuracy_score = exercise_state.accuracy_score
    
    # Process frame with PoseEstimator
    frame, data = exercise_state.estimator.detect_pose(frame, exercise_type)
    
    # Update exercise state
    exercise_state.counter = exercise_state.estimator.counter
    exercise_state.stage = exercise_state.estimator.stage
    exercise_state.perfect_reps = exercise_state.estimator.perfect_reps
    exercise_state.accuracy_score = exercise_state.estimator.accuracy_score
    exercise_state.form_feedback = exercise_state.estimator.form_feedback
    
    # Calculate total accuracy
    if exercise_state.counter > 0:
        exercise_state.total_accuracy = round((exercise_state.perfect_reps / exercise_state.counter) * 100, 2)
    
    return (
        frame,
        exercise_state.counter,
        exercise_state.stage,
        data.get('accuracy', 0),
        exercise_state.total_accuracy
    )

if __name__ == "__main__":
    test_pose_estimator()
