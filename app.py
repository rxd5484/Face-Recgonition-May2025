
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from emotion_detection import EmotionDetector
import cv2
from datetime import datetime, timedelta
import time
import os
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Emotion Detection Dashboard",
    page_icon="😀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4f8bf9;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #6c757d;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .emotion-label {
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the emotion detector
@st.cache_resource
def get_emotion_detector():
    return EmotionDetector()

detector = get_emotion_detector()

# Sidebar
st.sidebar.markdown("<div class='main-header'>Emotion Dashboard</div>", unsafe_allow_html=True)
st.sidebar.markdown("Analyze facial expressions and track emotions over time.")

# Mode selection
mode = st.sidebar.radio("Select Mode", ["Live Webcam", "Upload Image", "Upload Video", "View Historical Data"])

# Function to create emotion visualizations
def create_emotion_visualizations(df):
    # Filter out rows where no face was detected
    df_with_faces = df[df['face_detected'] == True].copy()
    
    if len(df_with_faces) == 0:
        st.warning("No faces were detected in this session.")
        return
    
    # 1. Create a time series of the dominant emotions
    st.markdown("<div class='sub-header'>Emotion Time Series</div>", unsafe_allow_html=True)
    
    # Convert timestamp to proper datetime if it's not already
    if isinstance(df_with_faces['timestamp'].iloc[0], str):
        df_with_faces['timestamp'] = pd.to_datetime(df_with_faces['timestamp'])
    
    # Get dominant emotion counts over time
    emotion_counts = df_with_faces.groupby([pd.Grouper(key='timestamp', freq='5S'), 'dominant_emotion']).size().reset_index(name='count')
    
    # Pivot the data to create a time series for each emotion
    emotion_time_series = emotion_counts.pivot(index='timestamp', columns='dominant_emotion', values='count').fillna(0)
    
    # Plot as line chart
    fig = px.line(emotion_time_series, x=emotion_time_series.index, y=emotion_time_series.columns,
                 title="Emotions Over Time",
                 labels={"value": "Count", "variable": "Emotion", "timestamp": "Time"},
                 color_discrete_sequence=px.colors.qualitative.Bold)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. Create a pie chart of dominant emotions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='sub-header'>Dominant Emotions</div>", unsafe_allow_html=True)
        dominant_emotions = df_with_faces['dominant_emotion'].value_counts()
        
        fig = px.pie(
            values=dominant_emotions.values,
            names=dominant_emotions.index,
            title="Distribution of Dominant Emotions",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # 3. Create a radar chart of average emotion scores
    with col2:
        st.markdown("<div class='sub-header'>Average Emotion Scores</div>", unsafe_allow_html=True)
        
        # Calculate average scores for each emotion
        emotion_columns = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        avg_emotions = df_with_faces[emotion_columns].mean().reset_index()
        avg_emotions.columns = ['emotion', 'score']
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=avg_emotions['score'],
            theta=avg_emotions['emotion'],
            fill='toself',
            line_color='rgba(79, 139, 249, 0.8)',
            fillcolor='rgba(79, 139, 249, 0.2)',
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            title="Average Emotion Scores"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 4. Create a heatmap of emotion transitions
    st.markdown("<div class='sub-header'>Emotion Transitions</div>", unsafe_allow_html=True)
    
    # Create a column with the next emotion
    df_with_faces['next_emotion'] = df_with_faces['dominant_emotion'].shift(-1)
    
    # Remove the last row (which has NaN for next_emotion)
    df_transitions = df_with_faces.dropna(subset=['next_emotion'])
    
    if len(df_transitions) > 1:  # Need at least 2 rows for transitions
        # Count transitions
        transition_counts = df_transitions.groupby(['dominant_emotion', 'next_emotion']).size().reset_index(name='count')
        
        # Pivot to create transition matrix
        transition_matrix = transition_counts.pivot(index='dominant_emotion', columns='next_emotion', values='count').fillna(0)
        
        # Normalize by row
        for idx in transition_matrix.index:
            row_sum = transition_matrix.loc[idx].sum()
            if row_sum > 0:
                transition_matrix.loc[idx] = transition_matrix.loc[idx] / row_sum
        
        # Create heatmap
        fig = px.imshow(
            transition_matrix,
            labels=dict(x="Next Emotion", y="Current Emotion", color="Probability"),
            x=transition_matrix.columns,
            y=transition_matrix.index,
            color_continuous_scale='Viridis',
            title="Emotion Transition Probabilities"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough emotion transitions detected to create a transition matrix.")
    
    # 5. Show stats in cards
    st.markdown("<div class='sub-header'>Emotion Statistics</div>", unsafe_allow_html=True)
    
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Total Detections", len(df_with_faces))
        st.markdown("</div>", unsafe_allow_html=True)
        
    with cols[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        most_common = dominant_emotions.index[0] if not dominant_emotions.empty else "None"
        st.metric("Most Common Emotion", most_common)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with cols[2]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        session_duration = (df_with_faces['timestamp'].max() - df_with_faces['timestamp'].min()).total_seconds()
        st.metric("Session Duration", f"{session_duration:.1f} sec")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with cols[3]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        unique_emotions = len(dominant_emotions)
        st.metric("Unique Emotions", unique_emotions)
        st.markdown("</div>", unsafe_allow_html=True)

# Function to process webcam feed
def process_webcam():
    st.markdown("<div class='main-header'>Live Webcam Emotion Detection</div>", unsafe_allow_html=True)
    
    # Session state to track if camera is running
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    
    # Button to start/stop the camera
    col1, col2 = st.columns([2, 1])
    
    with col1:
        button_text = "Stop Camera" if st.session_state.camera_running else "Start Camera"
        if st.button(button_text):
            st.session_state.camera_running = not st.session_state.camera_running
            if not st.session_state.camera_running:
                # Camera was stopped, save session data
                detector.save_session_data()
    
    with col2:
        process_every_n_frames = st.slider("Process every N frames", 1, 10, 3, 
                                           help="Higher values improve performance but might miss quick expressions")
    
    # Create a placeholder for the webcam feed
    webcam_placeholder = st.empty()
    
    # Dictionary to track emotion counts for the current session
    if 'emotion_counts' not in st.session_state:
        st.session_state.emotion_counts = {emotion: 0 for emotion in detector.emotions}
    
    # Create a bar chart placeholder
    chart_placeholder = st.empty()
    
    # Process the webcam feed
    if st.session_state.camera_running:
        try:
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            
            # Check if webcam opened successfully
            if not cap.isOpened():
                st.error("Could not open webcam. Please check your camera and try again.")
                st.session_state.camera_running = False
                return
            
            frame_count = 0
            start_time = time.time()
            
            while st.session_state.camera_running:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to capture image from webcam")
                    break
                
                frame_count += 1
                
                # Process every nth frame
                if frame_count % process_every_n_frames == 0:
                    # Detect emotions
                    result_frame, face_emotions = detector.detect_emotions(frame)
                    
                    # Convert BGR to RGB for display
                    result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                    
                    # Update emotion counts
                    for emotion, count in face_emotions.items():
                        st.session_state.emotion_counts[emotion] += count
                    
                    # Update the webcam feed
                    webcam_placeholder.image(result_rgb, channels="RGB", use_column_width=True)
                    
                    # Update the bar chart every 5 frames
                    if frame_count % (process_every_n_frames * 5) == 0:
                        # Create a bar chart of emotion counts
                        emotions = list(st.session_state.emotion_counts.keys())
                        counts = list(st.session_state.emotion_counts.values())
                        
                        if sum(counts) > 0:  # Only update if we have detected emotions
                            fig = px.bar(
                                x=emotions, 
                                y=counts,
                                color=emotions,
                                title="Detected Emotions",
                                labels={'x': 'Emotion', 'y': 'Count'},
                                color_discrete_sequence=px.colors.qualitative.Bold
                            )
                            chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Check if we should continue running
                if not st.session_state.camera_running:
                    break
                    
                # Small delay to reduce CPU usage
                time.sleep(0.01)
            
            # Release the webcam
            cap.release()
            
            # Save session data when camera stops
            detector.save_session_data()
            
            # Get the session file
            session_file = detector.session_file
            
            # Display results
            if os.path.exists(session_file):
                df = pd.read_csv(session_file, parse_dates=['timestamp'])
                create_emotion_visualizations(df)
            
        except Exception as e:
            st.error(f"Error processing webcam feed: {e}")
            st.session_state.camera_running = False
    else:
        webcam_placeholder.info("Click 'Start Camera' to begin emotion detection")

# Function to process uploaded image
def process_image():
    st.markdown("<div class='main-header'>Image Emotion Detection</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file = f"temp_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Process the image
            result_image, face_emotions, session_file = detector.process_image(temp_file)
            
            # Convert BGR to RGB for display
            result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            # Display the processed image
            st.image(result_rgb, caption="Processed Image", use_column_width=True)
            
            # Display emotion counts
            if face_emotions:
                st.markdown("<div class='sub-header'>Detected Emotions</div>", unsafe_allow_html=True)
                
                # Create a bar chart of emotion counts
                emotions = list(face_emotions.keys())
                counts = list(face_emotions.values())
                
                fig = px.bar(
                    x=emotions, 
                    y=counts,
                    color=emotions,
                    title="Detected Emotions",
                    labels={'x': 'Emotion', 'y': 'Count'},
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No faces detected in the image.")
            
            # Clean up the temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            # Display detailed visualizations if we have session data
            if session_file and os.path.exists(session_file):
                # Load the session data
                df = pd.read_csv(session_file, parse_dates=['timestamp'])
                
                # Create visualizations
                create_emotion_visualizations(df)
                
        except Exception as e:
            st.error(f"Error processing image: {e}")
            
            # Clean up the temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

# Function to process uploaded video
def process_video():
    st.markdown("<div class='main-header'>Video Emotion Detection</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file = f"temp_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process video settings
        process_every_n_frames = st.slider("Process every N frames", 1, 30, 10, 
                                          help="Higher values improve processing speed but might miss quick expressions")
        
        # Button to start processing
        if st.button("Process Video"):
            try:
                with st.spinner("Processing video (this may take a while)..."):
                    # Process the video
                    session_file = detector.process_video(temp_file, process_every_n_frames)
                
                st.success("Video processing complete!")
                
                # Clean up the temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
                # Display detailed visualizations
                if session_file and os.path.exists(session_file):
                    # Load the session data
                    df = pd.read_csv(session_file, parse_dates=['timestamp'])
                    
                    # Create visualizations
                    create_emotion_visualizations(df)
                    
            except Exception as e:
                st.error(f"Error processing video: {e}")
                
                # Clean up the temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)

# Function to view historical data
def view_historical_data():
    st.markdown("<div class='main-header'>Historical Emotion Data</div>", unsafe_allow_html=True)
    
    # Get all session files
    session_files = detector.get_all_sessions()
    
    if not session_files:
        st.info("No historical data found. Process some images or videos first.")
        return
    
    # Sort sessions by date (newest first)
    session_files.sort(reverse=True)
    
    # Let the user select a session
    selected_session = st.selectbox("Select Session", session_files)
    
    if selected_session:
        # Load the session data
        session_data = detector.load_session_data(selected_session)
        
        if session_data is not None:
            # Display the session details
            session_date = selected_session.split('_')[1].split('.')[0]
            session_time = selected_session.split('_')[2].split('.')[0]
            formatted_date = f"{session_date[:4]}-{session_date[4:6]}-{session_date[6:]}"
            formatted_time = f"{session_time[:2]}:{session_time[2:4]}:{session_time[4:]}"
            
            st.markdown(f"<div class='sub-header'>Session: {formatted_date} {formatted_time}</div>", unsafe_allow_html=True)
            
            # Create visualizations
            create_emotion_visualizations(session_data)
            
            # Option to download the data
            csv = session_data.to_csv(index=False)
            st.download_button(
                label="Download Session Data",
                data=csv,
                file_name=f"emotion_data_{formatted_date}_{formatted_time}.csv",
                mime="text/csv",
            )
        else:
            st.error(f"Error loading session data for {selected_session}")

# Main app logic
if mode == "Live Webcam":
    process_webcam()
elif mode == "Upload Image":
    process_image()
elif mode == "Upload Video":
    process_video()
elif mode == "View Historical Data":
    view_historical_data()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>Emotion Detection Dashboard | Built with OpenCV, FER, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
#streamlit run /Users/rakshitdongre/nvda_earnings/venv/app.py