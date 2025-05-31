import pandas as pd
import json
import os
from datetime import datetime

class DataManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.emotions_file = os.path.join(data_dir, "emotions.csv")
        self.session_file = os.path.join(data_dir, "session_data.json")
        self.ensure_data_directory()
        
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        os.makedirs(self.data_dir, exist_ok=True)
        
    def save_emotion_data(self, emotion_data_list):
        """Save emotion data to CSV"""
        if not emotion_data_list:
            return
            
        # Convert to DataFrame
        records = []
        for data in emotion_data_list:
            record = {'timestamp': data['timestamp']}
            record.update(data['emotions'])
            record['dominant_emotion'] = data['dominant_emotion']
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Append to existing file or create new
        if os.path.exists(self.emotions_file):
            existing_df = pd.read_csv(self.emotions_file)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(self.emotions_file, index=False)
    
    def load_emotion_data(self):
        """Load emotion data from CSV"""
        if os.path.exists(self.emotions_file):
            df = pd.read_csv(self.emotions_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return pd.DataFrame()
    
    def get_emotion_statistics(self, df):
        """Calculate emotion statistics"""
        if df.empty:
            return {}
        
        emotion_cols = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        stats = {}
        
        # Basic statistics
        stats['total_detections'] = len(df)
        stats['session_duration'] = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60
        
        # Emotion averages
        stats['avg_emotions'] = df[emotion_cols].mean().to_dict()
        
        # Dominant emotion distribution
        stats['emotion_distribution'] = df['dominant_emotion'].value_counts().to_dict()
        
        # Emotion transitions
        stats['transitions'] = self.calculate_emotion_transitions(df)
        
        return stats
    
    def calculate_emotion_transitions(self, df):
        """Calculate emotion transition probabilities"""
        if len(df) < 2:
            return {}
        
        transitions = {}
        for i in range(len(df) - 1):
            current = df.iloc[i]['dominant_emotion']
            next_emotion = df.iloc[i + 1]['dominant_emotion']
            
            if current not in transitions:
                transitions[current] = {}
            if next_emotion not in transitions[current]:
                transitions[current][next_emotion] = 0
            
            transitions[current][next_emotion] += 1
        
        # Convert to probabilities
        for current in transitions:
            total = sum(transitions[current].values())
            for next_emotion in transitions[current]:
                transitions[current][next_emotion] /= total
        
        return transitions