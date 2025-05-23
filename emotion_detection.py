import cv2
import numpy as np
from fer import FER
import pandas as pd
from datetime import datetime
import os
import json

class EmotionDetector:
    def __init__(self):
        # Initialize the FER detector with the default CNN model
        self.emotion_detector = FER(mtcnn=True)
        
        # Define the emotions we're tracking
        self.emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        
        # Create storage directory if it doesn't exist
        self.data_dir = "emotion_data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # Path to the session data file
        self.session_file = os.path.join(self.data_dir, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        # Initialize the session data
        self.session_data = []
        
        # Haarcascade for faster face detection (optional fallback)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_emotions(self, frame):
        """
        Detect emotions in a single frame.
        Returns: 
        - The frame with emotion labels 
        - Dictionary of emotions detected
        """
        # Make a copy of the frame to draw on
        result_frame = frame.copy()
        
        # Convert to RGB (FER expects RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect emotions
        detected_emotions = self.emotion_detector.detect_emotions(rgb_frame)
        
        # Current timestamp
        timestamp = datetime.now()
        
        # If no faces detected, return empty results
        if not detected_emotions:
            # Record that no face was found
            self.session_data.append({
                'timestamp': timestamp,
                'face_detected': False,
                'dominant_emotion': None,
                'angry': 0,
                'disgust': 0,
                'fear': 0,
                'happy': 0,
                'sad': 0,
                'surprise': 0,
                'neutral': 0
            })
            return result_frame, {}
        
        # Process each detected face
        face_emotions = {}
        
        for detection in detected_emotions:
            # Get face coordinates and emotions
            x, y, w, h = detection['box']
            emotions = detection['emotions']
            
            # Draw a rectangle around the face
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Find the dominant emotion
            dominant_emotion = max(emotions, key=emotions.get)
            face_emotions[dominant_emotion] = face_emotions.get(dominant_emotion, 0) + 1
            
            # Display the dominant emotion and its probability
            emotion_text = f"{dominant_emotion}: {emotions[dominant_emotion]:.2f}"
            cv2.putText(result_frame, emotion_text, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Record this detection in our session data
            record = {
                'timestamp': timestamp,
                'face_detected': True,
                'dominant_emotion': dominant_emotion,
                'angry': emotions['angry'],
                'disgust': emotions['disgust'],
                'fear': emotions['fear'],
                'happy': emotions['happy'],
                'sad': emotions['sad'],
                'surprise': emotions['surprise'],
                'neutral': emotions['neutral']
            }
            self.session_data.append(record)
            
        # Return the result
        return result_frame, face_emotions
    
    def process_video(self, video_source=0, process_every_n_frames=5):
        """
        Process video from a webcam or file and detect emotions.
        
        Args:
            video_source: Webcam index (usually 0) or path to video file
            process_every_n_frames: Only process every nth frame to improve performance
        """
        # Open the video source
        cap = cv2.VideoCapture(video_source)
        
        frame_count = 0
        running = True
        
        while running and cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
                
            frame_count += 1
            
            # Only process every nth frame for better performance
            if frame_count % process_every_n_frames == 0:
                # Detect emotions in the current frame
                result_frame, _ = self.detect_emotions(frame)
                
                # Display the result
                cv2.imshow('Emotion Detection', result_frame)
            else:
                # Display the original frame when not processing
                cv2.imshow('Emotion Detection', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Save the session data
        self.save_session_data()
        
        return self.session_file
    
    def process_image(self, image_path):
        """Process a single image and detect emotions."""
        # Read the image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error loading image: {image_path}")
            return None
        
        # Detect emotions
        result_image, face_emotions = self.detect_emotions(image)
        
        # Save the session data
        self.save_session_data()
        
        return result_image, face_emotions, self.session_file
    
    def save_session_data(self):
        """Save the current session data to a CSV file."""
        if self.session_data:
            df = pd.DataFrame(self.session_data)
            df.to_csv(self.session_file, index=False)
            print(f"Session data saved to {self.session_file}")
    
    def get_all_sessions(self):
        """Get a list of all saved session files."""
        if not os.path.exists(self.data_dir):
            return []
        
        session_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        return session_files
    
    def load_session_data(self, session_file):
        """Load data from a specific session file."""
        file_path = os.path.join(self.data_dir, session_file)
        if os.path.exists(file_path):
            return pd.read_csv(file_path, parse_dates=['timestamp'])
        return None

if __name__ == "__main__":
    # Test the emotion detector
    detector = EmotionDetector()
    
    # Use webcam (0) as the default source
    session_file = detector.process_video(0)
    print(f"Session saved to: {session_file}")