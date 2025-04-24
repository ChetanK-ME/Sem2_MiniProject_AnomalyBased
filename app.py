import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_preprocessing import DataPreprocessor
from model_training import AnomalyDetector
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Set page config
st.set_page_config(
    page_title="Cybersecurity Anomaly Detection",
    page_icon="🔒",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'y_score' not in st.session_state:
    st.session_state.y_score = None

def load_and_train_model():
    """Load data and train the model"""
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('cybersecurity_anomaly_dataset_10000.csv')
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df)
    
    # Train model
    detector = AnomalyDetector()
    best_model_name, train_metrics = detector.train_models(X_train, y_train)
    test_metrics = detector.evaluate_model(X_test, y_test)
    
    # Save model and preprocessor
    detector.save_model()
    preprocessor.save_scaler()
    
    # Get probability scores for ROC curve
    y_score = None
    if best_model_name == 'isolation_forest':
        y_score = -detector.best_model.score_samples(X_test)
        y_score = (y_score - np.min(y_score)) / (np.max(y_score) - np.min(y_score))
    elif best_model_name == 'one_class_svm':
        y_score = -detector.best_model.score_samples(X_test)
        y_score = (y_score - np.min(y_score)) / (np.max(y_score) - np.min(y_score))
    else:  # autoencoder
        y_prob = detector.best_model.predict_proba(X_test)
        y_score = y_prob[:, 1]
    
    return detector, preprocessor, train_metrics, test_metrics, df, y_test, y_score

def plot_metrics(metrics, title):
    """Plot evaluation metrics"""
    # Filter out confusion_matrix from metrics
    plot_metrics = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
    
    fig = go.Figure(data=[
        go.Bar(name='Score', x=list(plot_metrics.keys()), y=list(plot_metrics.values()))
    ])
    fig.update_layout(title=title, xaxis_title='Metric', yaxis_title='Score')
    return fig

def plot_model_comparison(metrics):
    """Plot comparison of all models"""
    models = list(metrics.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    
    data = []
    for model in models:
        for metric in metric_names:
            data.append({
                'Model': model,
                'Metric': metric,
                'Score': metrics[model][metric]
            })
    
    df = pd.DataFrame(data)
    fig = px.bar(df, x='Model', y='Score', color='Metric', barmode='group',
                 title='Model Performance Comparison')
    return fig

def plot_roc_curve(y_test, y_score):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC curve (AUC = {roc_auc:.2f})',
                             mode='lines'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random',
                             mode='lines', line=dict(dash='dash')))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=700
    )
    
    return fig

def plot_feature_importance(model, feature_names):
    """Plot feature importance for the best model"""
    if model.best_model_name == 'isolation_forest':
        importances = model.best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig = go.Figure(data=[
            go.Bar(x=[feature_names[i] for i in indices], y=importances[indices])
        ])
        fig.update_layout(title='Feature Importance (Isolation Forest)',
                         xaxis_title='Feature', yaxis_title='Importance')
        return fig
    else:
        return None

def main():
    st.title("🔒 Cybersecurity Anomaly Detection System")
    st.write("""
    This application uses machine learning to detect cybersecurity threats and anomalies in network traffic.
    The system implements multiple anomaly detection algorithms and automatically selects the best performing one.
    """)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Model Training", "Anomaly Detection", "Performance Analysis"])
    
    if page == "Model Training":
        st.header("Model Training")
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                detector, preprocessor, train_metrics, test_metrics, df, y_test, y_score = load_and_train_model()
                st.session_state.model = detector
                st.session_state.preprocessor = preprocessor
                st.session_state.metrics = train_metrics
                st.session_state.best_model_name = detector.best_model_name
                st.session_state.y_test = y_test
                st.session_state.y_score = y_score
                
                st.success(f"Model trained successfully! Best model: {detector.best_model_name}")
                
                # Display metrics for all models
                st.subheader("Model Performance Comparison")
                st.plotly_chart(plot_model_comparison(train_metrics))
                
                # Display best model metrics
                st.subheader(f"Best Model Metrics ({detector.best_model_name})")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_metrics(train_metrics[detector.best_model_name], "Training Metrics"))
                with col2:
                    st.plotly_chart(plot_metrics({k: v for k, v in test_metrics.items() if k != 'confusion_matrix'}, "Test Metrics"))
                
                # Display confusion matrix
                st.subheader("Confusion Matrix")
                cm = test_metrics['confusion_matrix']
                fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                              x=['Normal', 'Anomaly'], y=['Normal', 'Anomaly'])
                st.plotly_chart(fig)
                
                # Display ROC curve
                st.subheader("ROC Curve")
                st.plotly_chart(plot_roc_curve(y_test, y_score))
    
    elif page == "Anomaly Detection":
        st.header("Anomaly Detection")
        if st.session_state.model is None:
            st.warning("Please train the model first!")
            return
        
        st.subheader("Input Network Traffic Data")
        
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_data = {}
            input_data['src_port'] = st.number_input("Source Port (0-65535)", min_value=0, max_value=65535, value=0)
            input_data['dst_port'] = st.number_input("Destination Port (0-65535)", min_value=0, max_value=65535, value=0)
            
            # Create a dictionary mapping protocol numbers to their descriptions
            protocol_dict = {
                0: "TCP (Transmission Control Protocol)",
                1: "UDP (User Datagram Protocol)",
                2: "ICMP (Internet Control Message Protocol)",
                3: "HTTP (Hypertext Transfer Protocol)",
                4: "HTTPS (HTTP Secure)",
                5: "DNS (Domain Name System)",
                6: "Other"
            }
            
            # Create protocol options with descriptive labels
            protocol_options = [(num, f"{num} - {desc}") for num, desc in protocol_dict.items()]
            selected_protocol = st.selectbox(
                "Protocol",
                options=[num for num, _ in protocol_options],
                format_func=lambda x: protocol_dict[x],
                help="""Protocol Numbers and Their Descriptions:
                - 0: TCP - Reliable, connection-oriented protocol
                - 1: UDP - Fast, connectionless protocol
                - 2: ICMP - Network diagnostic and error reporting
                - 3: HTTP - Web protocol for unencrypted traffic
                - 4: HTTPS - Secure encrypted web protocol
                - 5: DNS - Domain name resolution protocol
                - 6: Other protocols"""
            )
            input_data['protocol'] = selected_protocol
            
            # Display protocol information box
            with st.expander("📚 Protocol Information"):
                st.markdown("""
                ### Network Protocols Overview
                
                #### TCP (0)
                - Connection-oriented protocol
                - Reliable data delivery
                - Used for: Web browsing, email, file transfer
                
                #### UDP (1)
                - Connectionless protocol
                - Fast, but unreliable
                - Used for: Streaming, gaming, DNS queries
                
                #### ICMP (2)
                - Network diagnostic protocol
                - Error reporting
                - Used for: Ping, traceroute
                
                #### HTTP (3)
                - Web protocol
                - Unencrypted data transfer
                - Default port: 80
                
                #### HTTPS (4)
                - Secure web protocol
                - Encrypted data transfer
                - Default port: 443
                
                #### DNS (5)
                - Domain name resolution
                - Converts domain names to IP addresses
                - Default port: 53
                
                #### Other (6)
                - Other network protocols
                - Examples: FTP, SSH, SMTP, etc.
                """)
            
            input_data['flow_duration'] = st.number_input("Flow Duration (ms)", min_value=0, value=0)
            input_data['pkt_count'] = st.number_input("Packet Count", min_value=0, value=0)

        with col2:
            input_data['pkt_size_avg'] = st.number_input("Average Packet Size (bytes)", min_value=0.0, value=0.0)
            input_data['bytes_sent'] = st.number_input("Bytes Sent", min_value=0, value=0)
            input_data['bytes_received'] = st.number_input("Bytes Received", min_value=0, value=0)
            input_data['conn_count_last_10s'] = st.number_input("Connection Count (Last 10s)", min_value=0, value=0)
            input_data['same_dst_count'] = st.number_input("Same Destination Count", min_value=0, value=0)

        with col3:
            input_data['srv_serror_rate'] = st.number_input("Service Error Rate", min_value=0.0, max_value=1.0, value=0.0)
            input_data['dst_host_srv_count'] = st.number_input("Destination Host Service Count", min_value=0, value=0)
            input_data['dst_host_same_srv_rate'] = st.number_input("Destination Host Same Service Rate", min_value=0.0, max_value=1.0, value=0.0)
            input_data['entropy'] = st.number_input("Traffic Entropy", min_value=0.0, value=0.0)
            input_data['honeypot_flag'] = st.selectbox("Honeypot Flag", options=[0, 1], help="0: Not from honeypot, 1: From honeypot")
        
        if st.button("Detect Anomaly"):
            # Convert input to numpy array in the correct order
            feature_order = ['src_port', 'dst_port', 'protocol', 'flow_duration', 'pkt_count',
                           'pkt_size_avg', 'bytes_sent', 'bytes_received', 'conn_count_last_10s',
                           'same_dst_count', 'srv_serror_rate', 'dst_host_srv_count',
                           'dst_host_same_srv_rate', 'entropy', 'honeypot_flag']
            
            input_array = np.array([[input_data[feature] for feature in feature_order]])
            
            # Preprocess input
            processed_input = st.session_state.preprocessor.transform_single_input(input_array)
            
            # Make prediction
            prediction = st.session_state.model.predict(processed_input)
            result = "Normal" if prediction[0] == 1 else "Anomaly"
            
            # Get probability scores
            prob_scores = st.session_state.model.predict_proba(processed_input)
            normal_prob = prob_scores[0, 0]
            anomaly_prob = prob_scores[0, 1]
            
            # Create columns for results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                # Display prediction
                if result == "Normal":
                    st.success(f"✅ This traffic pattern is classified as {result}")
                else:
                    st.error(f"⚠️ This traffic pattern is classified as {result}")
                
                # Display probability scores
                st.info(f"Probability of being normal: {normal_prob:.4f}")
                st.info(f"Probability of being anomaly: {anomaly_prob:.4f}")
            
            with result_col2:
                # Create gauge chart for anomaly probability
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=anomaly_prob * 100,
                    title={'text': "Anomaly Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 75], 'color': "yellow"},
                            {'range': [75, 100], 'color': "salmon"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 75
                        }
                    }
                ))
                st.plotly_chart(fig)
            
            # If using autoencoder, show reconstruction error
            if st.session_state.best_model_name == 'autoencoder':
                reconstructed = st.session_state.model.best_model.model.predict(processed_input)
                mse = np.mean(np.power(processed_input - reconstructed, 2))
                st.info(f"Reconstruction Error: {mse:.4f}")
                
            # Display feature values as a bar chart
            st.subheader("Input Feature Values")
            feature_df = pd.DataFrame({
                'Feature': feature_order,
                'Value': input_array[0]
            })
            fig = px.bar(feature_df, x='Value', y='Feature', orientation='h',
                        title='Network Traffic Features',
                        labels={'Value': 'Feature Value', 'Feature': 'Feature Name'})
            fig.update_layout(showlegend=False, height=600)
            st.plotly_chart(fig)
    
    else:  # Performance Analysis
        st.header("Performance Analysis")
        if st.session_state.model is None:
            st.warning("Please train the model first!")
            return
        
        # Model comparison
        st.subheader("Model Comparison")
        st.plotly_chart(plot_model_comparison(st.session_state.metrics))
        
        # ROC curve
        st.subheader("ROC Curve")
        st.plotly_chart(plot_roc_curve(st.session_state.y_test, st.session_state.y_score))
        
        # Feature importance (if available)
        if st.session_state.model.best_model_name == 'isolation_forest':
            st.subheader("Feature Importance")
            feature_names = [f'Feature {i}' for i in range(10)]  # Update with actual feature names
            fig = plot_feature_importance(st.session_state.model, feature_names)
            if fig:
                st.plotly_chart(fig)
        
        # Algorithm-specific visualizations
        st.subheader("Algorithm-Specific Analysis")
        if st.session_state.best_model_name == 'autoencoder':
            st.write("Autoencoder Architecture:")
            st.write("- Input Layer: 10 neurons")
            st.write("- Encoder: 32 neurons with ReLU activation")
            st.write("- Decoder: 10 neurons with Sigmoid activation")
            st.write("- Dropout: 0.2 for regularization")
        
        elif st.session_state.best_model_name == 'isolation_forest':
            st.write("Isolation Forest Parameters:")
            st.write("- Contamination: 0.1")
            st.write("- Random State: 42")
            st.write("- Number of Trees: 100 (default)")
        
        elif st.session_state.best_model_name == 'one_class_svm':
            st.write("One-Class SVM Parameters:")
            st.write("- Kernel: RBF")
            st.write("- Nu: 0.1")
            st.write("- Gamma: scale (default)")

    # Add explanation section
    st.markdown("""
    ### Feature Descriptions:
    - **Source Port**: The port number of the source system (0-65535)
    - **Destination Port**: The port number of the destination system (0-65535)
    - **Protocol**: The network protocol used (TCP, UDP, ICMP, etc.)
    - **Flow Duration**: The duration of the network flow in milliseconds
    - **Packet Count**: Total number of packets in the flow
    - **Average Packet Size**: Average size of packets in bytes
    - **Bytes Sent**: Total bytes sent in the flow
    - **Bytes Received**: Total bytes received in the flow
    - **Connection Count (Last 10s)**: Number of connections made in the last 10 seconds
    - **Same Destination Count**: Number of connections to the same destination
    - **Service Error Rate**: Rate of service errors
    - **Destination Host Service Count**: Number of services running on the destination host
    - **Destination Host Same Service Rate**: Rate of same service connections to the destination
    - **Entropy**: Measure of randomness in the traffic
    - **Honeypot Flag**: Indicates if the traffic was captured by a honeypot
    """)

if __name__ == "__main__":
    main() 