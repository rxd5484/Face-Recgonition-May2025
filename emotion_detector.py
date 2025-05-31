import cv2
import numpy as np
from deepface import DeepFace
import pandas as pd
import time
from datetime import datetime
import threading
import queue

class EmotionDetector:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.emotion_queue = queue.Queue()
        self.current_frame = None
        self.current_emotions = None
        
        # Initialize face cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def initialize_camera(self):
        """Initialize webcam capture"""
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def detect_emotions(self, frame):
        """Detect emotions in a single frame using DeepFace"""
        try:
            # Convert BGR to RGB for DeepFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces first
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Get the largest face
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                x, y, w, h = largest_face
                
                # Extract face region
                face_roi = rgb_frame[y:y+h, x:x+w]
                
                if face_roi.size > 0:
                    # Analyze emotions using DeepFace
                    try:
                        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                        
                        # Handle both single result and list of results
                        if isinstance(result, list):
                            result = result[0]
                        
                        emotions = result['emotion']
                        
                        # Normalize emotion names to match FER format
                        emotion_mapping = {
                            'angry': 'angry',
                            'disgust': 'disgust', 
                            'fear': 'fear',
                            'happy': 'happy',
                            'sad': 'sad',
                            'surprise': 'surprise',
                            'neutral': 'neutral'
                        }
                        
                        # Convert percentages to probabilities (0-1 range)
                        normalized_emotions = {}
                        for emotion, value in emotions.items():
                            mapped_emotion = emotion_mapping.get(emotion.lower(), emotion.lower())
                            normalized_emotions[mapped_emotion] = value / 100.0
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Get dominant emotion
                        dominant_emotion = max(normalized_emotions, key=normalized_emotions.get)
                        confidence = normalized_emotions[dominant_emotion]
                        
                        # Display emotion on frame
                        cv2.putText(frame, f"{dominant_emotion}: {confidence:.2f}", 
                                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        return normalized_emotions, frame
                        
                    except Exception as e:
                        print(f"DeepFace analysis error: {e}")
                        # Return default emotions if analysis fails
                        default_emotions = {
                            'angry': 0.0, 'disgust': 0.0, 'fear': 0.0, 'happy': 0.0,
                            'sad': 0.0, 'surprise': 0.0, 'neutral': 1.0
                        }
                        return default_emotions, frame
            
            return None, frame
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return None, frame
    
    def start_detection(self):
        """Start real-time emotion detection"""
        if not self.initialize_camera():
            return False
            
        self.is_running = True
        detection_thread = threading.Thread(target=self._detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        return True
    
    def _detection_loop(self):
        """Main detection loop running in separate thread"""
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Detect emotions
            emotions, processed_frame = self.detect_emotions(frame.copy())
            
            # Update current data
            self.current_frame = processed_frame
            self.current_emotions = emotions
            
            # Add to queue with timestamp
            if emotions:
                emotion_data = {
                    'timestamp': datetime.now(),
                    'emotions': emotions,
                    'dominant_emotion': max(emotions, key=emotions.get)
                }
                self.emotion_queue.put(emotion_data)
            
            time.sleep(0.1)  # Control frame rate
    
    def get_current_frame(self):
        """Get current processed frame"""
        return self.current_frame
    
    def get_current_emotions(self):
        """Get current emotion data"""
        return self.current_emotions
    
    def get_emotion_data(self):
        """Get all queued emotion data"""
        data = []
        while not self.emotion_queue.empty():
            try:
                data.append(self.emotion_queue.get_nowait())
            except queue.Empty:
                break
        return data
    
    def stop_detection(self):
        """Stop emotion detection"""
        self.is_running = False
        if self.cap:
            self.cap.release()
