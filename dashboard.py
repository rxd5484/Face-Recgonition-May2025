import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class Dashboard:
    def __init__(self):
        self.emotion_colors = {
            'happy': '#FFD700',
            'sad': '#4169E1',
            'angry': '#FF4500',
            'fear': '#8A2BE2',
            'surprise': '#FF69B4',
            'disgust': '#32CD32',
            'neutral': '#808080'
        }
    
    def create_realtime_line_chart(self, df):
        """Create real-time emotion trends line chart"""
        if df.empty:
            return go.Figure()
        
        emotion_cols = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        fig = go.Figure()
        
        for emotion in emotion_cols:
            if emotion in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df[emotion],
                    mode='lines+markers',
                    name=emotion.capitalize(),
                    line=dict(color=self.emotion_colors.get(emotion, '#000000')),
                    hovertemplate=f'<b>{emotion.capitalize()}</b><br>' +
                                  'Time: %{x}<br>' +
                                  'Confidence: %{y:.3f}<extra></extra>'
                ))
        
        fig.update_layout(
            title="Real-Time Emotion Trends",
            xaxis_title="Time",
            yaxis_title="Confidence Score",
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def create_emotion_pie_chart(self, stats):
        """Create emotion distribution pie chart"""
        if not stats or 'emotion_distribution' not in stats:
            return go.Figure()
        
        emotions = list(stats['emotion_distribution'].keys())
        values = list(stats['emotion_distribution'].values())
        colors = [self.emotion_colors.get(emotion.lower(), '#000000') for emotion in emotions]
        
        fig = go.Figure(data=[go.Pie(
            labels=emotions,
            values=values,
            marker_colors=colors,
            hovertemplate='<b>%{label}</b><br>' +
                          'Count: %{value}<br>' +
                          'Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Emotion Distribution",
            height=400
        )
        
        return fig
    
    def create_radar_chart(self, stats):
        """Create radar chart for average emotions"""
        if not stats or 'avg_emotions' not in stats:
            return go.Figure()
        
        emotions = list(stats['avg_emotions'].keys())
        values = list(stats['avg_emotions'].values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=emotions,
            fill='toself',
            name='Average Emotions',
            line_color='rgb(32, 201, 151)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.1] if values else [0, 1]
                )),
            title="Emotion Intensity Radar",
            height=400
        )
        
        return fig
    
    def create_transition_heatmap(self, stats):
        """Create emotion transition heatmap"""
        if not stats or 'transitions' not in stats or not stats['transitions']:
            return go.Figure()
        
        transitions = stats['transitions']
        emotions = list(set(list(transitions.keys()) + 
                           [e for trans in transitions.values() for e in trans.keys()]))
        
        # Create transition matrix
        matrix = np.zeros((len(emotions), len(emotions)))
        emotion_to_idx = {emotion: i for i, emotion in enumerate(emotions)}
        
        for from_emotion, to_emotions in transitions.items():
            from_idx = emotion_to_idx[from_emotion]
            for to_emotion, prob in to_emotions.items():
                to_idx = emotion_to_idx[to_emotion]
                matrix[from_idx][to_idx] = prob
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=emotions,
            y=emotions,
            colorscale='Viridis',
            hovertemplate='From: %{y}<br>To: %{x}<br>Probability: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Emotion Transition Probabilities",
            xaxis_title="To Emotion",
            yaxis_title="From Emotion",
            height=400
        )
        
        return fig
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class Dashboard:
    def __init__(self):
        self.emotion_colors = {
            'happy': '#FFD700',
            'sad': '#4169E1',
            'angry': '#FF4500',
            'fear': '#8A2BE2',
            'surprise': '#FF69B4',
            'disgust': '#32CD32',
            'neutral': '#808080'
        }
    
    def create_realtime_line_chart(self, df):
        """Create real-time emotion trends line chart"""
        if df.empty:
            return go.Figure()
        
        emotion_cols = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        fig = go.Figure()
        
        for emotion in emotion_cols:
            if emotion in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df[emotion],
                    mode='lines+markers',
                    name=emotion.capitalize(),
                    line=dict(color=self.emotion_colors.get(emotion, '#000000')),
                    hovertemplate=f'<b>{emotion.capitalize()}</b><br>' +
                                  'Time: %{x}<br>' +
                                  'Confidence: %{y:.3f}<extra></extra>'
                ))
        
        fig.update_layout(
            title="Real-Time Emotion Trends",
            xaxis_title="Time",
            yaxis_title="Confidence Score",
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def create_emotion_pie_chart(self, stats):
        """Create emotion distribution pie chart"""
        if not stats or 'emotion_distribution' not in stats:
            return go.Figure()
        
        emotions = list(stats['emotion_distribution'].keys())
        values = list(stats['emotion_distribution'].values())
        colors = [self.emotion_colors.get(emotion.lower(), '#000000') for emotion in emotions]
        
        fig = go.Figure(data=[go.Pie(
            labels=emotions,
            values=values,
            marker_colors=colors,
            hovertemplate='<b>%{label}</b><br>' +
                          'Count: %{value}<br>' +
                          'Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Emotion Distribution",
            height=400
        )
        
        return fig
    
    def create_radar_chart(self, stats):
        """Create radar chart for average emotions"""
        if not stats or 'avg_emotions' not in stats:
            return go.Figure()
        
        emotions = list(stats['avg_emotions'].keys())
        values = list(stats['avg_emotions'].values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=emotions,
            fill='toself',
            name='Average Emotions',
            line_color='rgb(32, 201, 151)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.1] if values else [0, 1]
                )),
            title="Emotion Intensity Radar",
            height=400
        )
        
        return fig
    
    def create_transition_heatmap(self, stats):
        """Create emotion transition heatmap"""
        if not stats or 'transitions' not in stats or not stats['transitions']:
            return go.Figure()
        
        transitions = stats['transitions']
        emotions = list(set(list(transitions.keys()) + 
                           [e for trans in transitions.values() for e in trans.keys()]))
        
        # Create transition matrix
        matrix = np.zeros((len(emotions), len(emotions)))
        emotion_to_idx = {emotion: i for i, emotion in enumerate(emotions)}
        
        for from_emotion, to_emotions in transitions.items():
            from_idx = emotion_to_idx[from_emotion]
            for to_emotion, prob in to_emotions.items():
                to_idx = emotion_to_idx[to_emotion]
                matrix[from_idx][to_idx] = prob
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=emotions,
            y=emotions,
            colorscale='Viridis',
            hovertemplate='From: %{y}<br>To: %{x}<br>Probability: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Emotion Transition Probabilities",
            xaxis_title="To Emotion",
            yaxis_title="From Emotion",
            height=400
        )
        
        return fig