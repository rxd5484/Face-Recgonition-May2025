
import streamlit as st
import cv2
import time
from emotion_detector import EmotionDetector
from data_manager import DataManager
from dashboard import Dashboard
import threading

# Page configuration
st.set_page_config(
    page_title="Real-Time Emotion Detection Dashboard",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = EmotionDetector()
    st.session_state.data_manager = DataManager()
    st.session_state.dashboard = Dashboard()
    st.session_state.is_detecting = False
    st.session_state.emotion_buffer = []

def main():
    st.title("🎭 Real-Time Emotion Detection Dashboard")
    st.markdown("Built with Python, OpenCV, FER, and Streamlit")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎮 Controls")
        
        if st.button("🎥 Start Detection", key="start"):
            if not st.session_state.is_detecting:
                if st.session_state.detector.start_detection():
                    st.session_state.is_detecting = True
                    st.success("Detection started!")
                else:
                    st.error("Failed to start camera")
        
        if st.button("⏹️ Stop Detection", key="stop"):
            if st.session_state.is_detecting:
                st.session_state.detector.stop_detection()
                st.session_state.is_detecting = False
                st.success("Detection stopped!")
        
        st.markdown("---")
        st.header("📊 Settings")
        auto_save = st.checkbox("Auto-save data", value=True)
        refresh_rate = st.slider("Refresh rate (seconds)", 0.5, 3.0, 1.0, 0.5)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📹 Live Feed")
        video_placeholder = st.empty()
        
        st.header("🎯 Current Emotions")
        current_emotions_placeholder = st.empty()
    
    with col2:
        st.header("📈 Real-Time Analytics")
        
        # Placeholder containers for charts
        line_chart_placeholder = st.empty()
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            pie_chart_placeholder = st.empty()
        with col2_2:
            radar_chart_placeholder = st.empty()
        
        heatmap_placeholder = st.empty()
    
    # Statistics section
    st.header("📊 Session Statistics")
    stats_placeholder = st.empty()
    
    # Real-time update loop
    if st.session_state.is_detecting:
        # Update data
        new_data = st.session_state.detector.get_emotion_data()
        if new_data:
            st.session_state.emotion_buffer.extend(new_data)
            
            # Auto-save if enabled
            if auto_save and len(st.session_state.emotion_buffer) >= 10:
                st.session_state.data_manager.save_emotion_data(st.session_state.emotion_buffer)
                st.session_state.emotion_buffer = []
        
        # Update video feed
        current_frame = st.session_state.detector.get_current_frame()
        if current_frame is not None:
            video_placeholder.image(current_frame, channels="BGR", width=640)
        
        # Update current emotions
        current_emotions = st.session_state.detector.get_current_emotions()
        if current_emotions:
            with current_emotions_placeholder.container():
                for emotion, confidence in current_emotions.items():
                    st.metric(
                        label=emotion.capitalize(),
                        value=f"{confidence:.3f}",
                        delta=None
                    )
    
    # Load and display analytics
    df = st.session_state.data_manager.load_emotion_data()
    if not df.empty:
        # Limit to recent data for performance
        recent_df = df.tail(500)
        stats = st.session_state.data_manager.get_emotion_statistics(recent_df)
        
        # Update charts
        dashboard = st.session_state.dashboard
        
        with line_chart_placeholder:
            fig_line = dashboard.create_realtime_line_chart(recent_df.tail(100))
            st.plotly_chart(fig_line, use_container_width=True, key="line_chart")
        
        with pie_chart_placeholder:
            fig_pie = dashboard.create_emotion_pie_chart(stats)
            st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart")
        
        with radar_chart_placeholder:
            fig_radar = dashboard.create_radar_chart(stats)
            st.plotly_chart(fig_radar, use_container_width=True, key="radar_chart")
        
        with heatmap_placeholder:
            fig_heatmap = dashboard.create_transition_heatmap(stats)
            st.plotly_chart(fig_heatmap, use_container_width=True, key="heatmap")
        
        # Display statistics
        with stats_placeholder:
            if stats:
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                
                with col_s1:
                    st.metric("Total Detections", stats.get('total_detections', 0))
                
                with col_s2:
                    duration = stats.get('session_duration', 0)
                    st.metric("Session Duration", f"{duration:.1f} min")
                
                with col_s3:
                    if 'emotion_distribution' in stats and stats['emotion_distribution']:
                        dominant = max(stats['emotion_distribution'], 
                                     key=stats['emotion_distribution'].get)
                        st.metric("Dominant Emotion", dominant.capitalize())
                
                with col_s4:
                    if 'avg_emotions' in stats and stats['avg_emotions']:
                        avg_happiness = stats['avg_emotions'].get('happy', 0)
                        st.metric("Avg Happiness", f"{avg_happiness:.3f}")
    
    # Auto-refresh
    if st.session_state.is_detecting:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()
