# wine_streamlit_app.py - Enhanced Professional Wine Quality Classifier
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="Wine Quality Classifier",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    /* Main layout improvements */
    .main-header {
        font-size: 2.5rem;
        color: #722F37;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .section-header {
        font-size: 1.4rem;
        color: #722F37;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
        border-left: 4px solid #722F37;
        padding-left: 1rem;
    }
    
    .subsection-header {
        font-size: 1.1rem;
        color: #555;
        font-weight: 500;
        margin: 0.8rem 0 0.3rem 0;
    }
    
    /* Enhanced prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #f0f2f6 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid #722F37;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        margin: 1rem 0;
    }
    
    /* Professional metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #dee2e6;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Dashboard enhancements */
    .dashboard-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border-top: 3px solid #722F37;
        margin: 1rem 0;
    }
    
    .dashboard-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f1f3f4;
    }
    
    .dashboard-icon {
        font-size: 1.8rem;
        color: #722F37;
        margin-right: 1rem;
    }
    
    .dashboard-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
    }
    
    /* Status indicators */
    .status-success { color: #28a745; font-weight: 600; }
    .status-warning { color: #ffc107; font-weight: 600; }
    .status-error { color: #dc3545; font-weight: 600; }
    
    /* Improved spacing */
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        background: linear-gradient(45deg, #722F37, #8b4513);
        color: white;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #8b4513, #722F37);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(114, 47, 55, 0.3);
    }
    
    /* Remove default streamlit margins */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_components():
    """Load ALL model components with comprehensive error handling"""
    required_files = [
        'wine_model.pkl',
        'wine_scaler.pkl', 
        'wine_label_encoder.pkl',
        'wine_feature_names.pkl',
        'wine_type_encoder.pkl',
        'feature_ranges.pkl',
        'dataset_stats.pkl',
        'model_performance.pkl'
    ]
    
    # Check if all files exist first
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        return tuple([None] * 8 + [False, missing_files])
    
    try:
        # Load core model components with explicit file handling
        with open('wine_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('wine_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('wine_label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open('wine_feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        with open('wine_type_encoder.pkl', 'rb') as f:
            wine_type_encoder = pickle.load(f)
        
        # Load metadata from actual training
        with open('feature_ranges.pkl', 'rb') as f:
            feature_ranges = pickle.load(f)
        
        with open('dataset_stats.pkl', 'rb') as f:
            dataset_stats = pickle.load(f)
        
        with open('model_performance.pkl', 'rb') as f:
            model_performance = pickle.load(f)
        
        # Validate loaded data
        if not all([model, scaler, label_encoder, feature_names, wine_type_encoder]):
            raise ValueError("One or more core components failed to load properly")
        
        return (model, scaler, label_encoder, feature_names, wine_type_encoder, 
                feature_ranges, dataset_stats, model_performance, True, [])
                
    except Exception as e:
        return tuple([None] * 8 + [False, [f"Loading error: {str(e)}"]])

def safe_create_slider(feature, feature_ranges, feature_labels, feature_help):
    """Safely create a slider with proper error handling"""
    try:
        if feature not in feature_ranges:
            # Provide reasonable defaults based on feature type
            if 'acidity' in feature or 'ph' in feature:
                ranges = {'min': 0.0, 'max': 15.0, 'mean': 3.0, 'std': 1.0}
            elif 'alcohol' in feature:
                ranges = {'min': 8.0, 'max': 15.0, 'mean': 11.0, 'std': 1.0}
            elif 'sugar' in feature:
                ranges = {'min': 0.0, 'max': 50.0, 'mean': 5.0, 'std': 3.0}
            elif 'sulfur' in feature:
                ranges = {'min': 0.0, 'max': 300.0, 'mean': 50.0, 'std': 20.0}
            elif 'density' in feature:
                ranges = {'min': 0.990, 'max': 1.010, 'mean': 0.997, 'std': 0.003}
            else:
                ranges = {'min': 0.0, 'max': 10.0, 'mean': 1.0, 'std': 0.5}
        else:
            ranges = feature_ranges[feature]
        
        # Get appropriate step size based on feature type
        if 'sulfur_dioxide' in feature:
            step_size = 1.0
            value_type = float
        elif feature == 'density':
            step_size = 0.0001
            value_type = float
        elif 'ph' in feature:
            step_size = 0.01
            value_type = float
        else:
            step_size = 0.01 if ranges['max'] - ranges['min'] < 10 else 0.1
            value_type = float
        
        slider_value = st.slider(
            feature_labels.get(feature, feature.replace('_', ' ').title()),
            min_value=value_type(ranges['min']),
            max_value=value_type(ranges['max']),
            value=value_type(ranges['mean']),
            step=step_size,
            help=f"{feature_help.get(feature, 'Wine property')} (Range: {ranges['min']:.3f} - {ranges['max']:.3f})"
        )
        
        return float(slider_value)
        
    except Exception as e:
        st.error(f"❌ Error creating input for {feature}: {e}")
        return 1.0

def make_prediction(input_values, model, scaler, label_encoder, feature_names):
    """Make prediction using the EXACT same pipeline as training"""
    try:
        # Create DataFrame with feature names from training
        feature_df = pd.DataFrame([input_values], columns=feature_names)
        
        # Apply the SAME scaling transformation used during training
        features_scaled = scaler.transform(feature_df)
        
        # Get prediction using the trained model
        prediction_encoded = model.predict(features_scaled)[0]
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
        else:
            # For models without probability support
            probabilities = np.zeros(len(label_encoder.classes_))
            probabilities[prediction_encoded] = 1.0
        
        # Convert back to original quality label
        quality_label = label_encoder.inverse_transform([prediction_encoded])[0]
        
        return quality_label, probabilities, True
        
    except Exception as e:
        return None, None, False

def display_wine_metrics_professional(user_inputs, wine_type):
    """Professional metrics display using enhanced Streamlit components"""
    st.markdown('<div class="subsection-header">🔍 Wine Analysis Summary</div>', unsafe_allow_html=True)
    
    # Use expandable sections instead of nested columns to avoid nesting issues
    with st.expander("🍷 Basic Properties", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Wine Type",
                value=wine_type.title(),
                help="Type of wine being analyzed"
            )
        with col2:
            st.metric(
                label="Alcohol Content",
                value=f"{user_inputs.get('alcohol', 0):.1f}%",
                help="Alcohol percentage by volume"
            )
        with col3:
            st.metric(
                label="Residual Sugar", 
                value=f"{user_inputs.get('residual_sugar', 0):.1f} g/L",
                help="Remaining sugar after fermentation"
            )
    
    with st.expander("🧪 Chemical Properties", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="pH Level",
                value=f"{user_inputs.get('ph', 0):.2f}",
                help="Acidity level of the wine"
            )
        with col2:
            st.metric(
                label="Volatile Acidity",
                value=f"{user_inputs.get('volatile_acidity', 0):.3f} g/L", 
                help="Acetic acid content"
            )
        with col3:
            st.metric(
                label="Fixed Acidity",
                value=f"{user_inputs.get('fixed_acidity', 0):.2f} g/L",
                help="Tartaric acid content"
            )
    
    with st.expander("⚗️ Preservatives & Physical", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Total SO₂",
                value=f"{user_inputs.get('total_sulfur_dioxide', 0):.0f} mg/L",
                help="Total sulfur dioxide content"
            )
        with col2:
            st.metric(
                label="Density",
                value=f"{user_inputs.get('density', 0):.4f} g/cm³",
                help="Wine density"
            )
        with col3:
            st.metric(
                label="Sulphates",
                value=f"{user_inputs.get('sulphates', 0):.2f} g/L",
                help="Wine preservative content"
            )

def display_model_confidence(probabilities, label_encoder):
    """Display model confidence with enhanced visualization"""
    st.markdown('<div class="subsection-header">📊 Model Confidence Analysis</div>', unsafe_allow_html=True)
    
    conf_df = pd.DataFrame({
        'Quality Level': label_encoder.classes_,
        'Confidence': probabilities * 100
    })
    
    # Enhanced confidence chart
    color_map = {
        'Poor': '#ff4b4b',
        'Average': '#ff8f00', 
        'Good': '#2e7d32'
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            conf_df, 
            x='Quality Level', 
            y='Confidence',
            title="🎯 Model Confidence Distribution",
            text='Confidence',
            color='Quality Level',
            color_discrete_map=color_map
        )
        
        fig.update_traces(
            texttemplate='%{text:.1f}%', 
            textposition='outside',
            textfont_size=12,
            textfont_color='black'
        )
        
        fig.update_layout(
            showlegend=False, 
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis=dict(
                title='Confidence (%)',
                range=[0, max(conf_df['Confidence']) * 1.2],
                showgrid=True, 
                gridwidth=1, 
                gridcolor='rgba(128,128,128,0.2)'
            ),
            xaxis=dict(
                title='Quality Level',
                showgrid=False
            ),
            title=dict(
                text="🎯 Model Confidence Distribution",
                x=0.5,
                font=dict(size=16, color='#722F37')
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**🎯 Confidence Breakdown:**")
        for _, row in conf_df.iterrows():
            quality = row['Quality Level']
            confidence = row['Confidence']
            color = color_map.get(quality, '#666666')
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {color}15 0%, {color}05 100%);
                border-left: 4px solid {color};
                padding: 1rem;
                margin: 0.5rem 0;
                border-radius: 8px;
            ">
                <div style="font-weight: 600; color: {color};">{quality} Quality</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {color};">{confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

def display_model_info_performance(dataset_stats, model_performance):
    """Display model information and performance in dashboard style"""
    st.markdown('<div class="section-header">📋 Model Information & Performance Overview</div>', unsafe_allow_html=True)
    
    # Professional information cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="dashboard-header">
                <span class="dashboard-icon">🤖</span>
                <div class="dashboard-title">Selected Model</div>
            </div>
            <p><strong>Algorithm:</strong> {dataset_stats['best_model_name']}</p>
            <p><strong>Type:</strong> {dataset_stats['best_model_type']}</p>
            <p><strong>Test Accuracy:</strong> <span class="status-success">{dataset_stats['best_model_accuracy']:.1%}</span></p>
            <p><strong>Features Used:</strong> {dataset_stats['feature_count']} properties</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        quality_dist = dataset_stats['quality_distribution']
        st.markdown(f"""
        <div class="metric-card">
            <div class="dashboard-header">
                <span class="dashboard-icon">📊</span>
                <div class="dashboard-title">Training Dataset</div>
            </div>
            <p><strong>Total Samples:</strong> {dataset_stats['total_samples']:,}</p>
            <p><strong>Red Wines:</strong> {dataset_stats['red_wines']:,}</p>
            <p><strong>White Wines:</strong> {dataset_stats['white_wines']:,}</p>
            <p><strong>Validation:</strong> 5-fold Cross-Validation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="dashboard-header">
                <span class="dashboard-icon">🎯</span>
                <div class="dashboard-title">Quality Classes</div>
            </div>
            <p><strong>Poor:</strong> {quality_dist.get('Poor', 0):,} samples</p>
            <p><strong>Average:</strong> {quality_dist.get('Average', 0):,} samples</p>
            <p><strong>Good:</strong> {quality_dist.get('Good', 0):,} samples</p>
            <p><strong>Source:</strong> Kaggle Wine Quality Dataset</p>
        </div>
        """, unsafe_allow_html=True)

def display_top_performing_models(model_performance):
    """Display top performing models section"""
    st.markdown('<div class="subsection-header">📈 Top Performing Models Ranking</div>', unsafe_allow_html=True)
    
    test_accs = model_performance['test_accuracies']
    top_models = sorted(test_accs.items(), key=lambda x: x[1], reverse=True)[:8]
    
    perf_df = pd.DataFrame(top_models, columns=['Model', 'Test Accuracy'])
    perf_df['Test Accuracy %'] = perf_df['Test Accuracy'].apply(lambda x: f"{x:.1%}")
    perf_df['Rank'] = range(1, len(perf_df) + 1)
    
    # Reorder columns
    perf_df_display = perf_df[['Rank', 'Model', 'Test Accuracy %']].copy()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(
            perf_df_display, 
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("🏆 Rank", width="small"),
                "Model": st.column_config.TextColumn("🤖 Model Algorithm", width="large"),
                "Test Accuracy %": st.column_config.TextColumn("📊 Accuracy", width="medium")
            }
        )
    
    with col2:
        # Create performance comparison chart
        fig = px.bar(
            perf_df.head(6), 
            x='Test Accuracy', 
            y='Model',
            orientation='h',
            title="🏆 Top 6 Models Performance",
            color='Test Accuracy',
            color_continuous_scale='viridis'
        )
        
        fig.update_traces(
            texttemplate='%{x:.1%}', 
            textposition='inside',
            textfont_color='white'
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis=dict(tickformat='.1%', title='Test Accuracy'),
            yaxis=dict(title='Models'),
            title_font=dict(size=14, color='#722F37'),
            plot_bgcolor='rgba(248,249,250,0.8)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def get_safe_colors():
    """Return safe color palettes for charts to avoid Plotly color format errors"""
    return {
        'solid_colors': ['#722F37', '#8b4513', '#a0522d', '#cd853f', '#daa520'],
        'fill_colors': [
            'rgba(114,47,55,0.25)',   # Dark red with transparency
            'rgba(139,69,19,0.25)',   # Brown with transparency  
            'rgba(160,82,45,0.25)',   # Light brown with transparency
            'rgba(205,133,63,0.25)',  # Peru with transparency
            'rgba(218,165,32,0.25)'   # Goldenrod with transparency
        ],
        'quality_colors': {
            'Poor': '#ff4b4b',
            'Average': '#ff8f00', 
            'Good': '#2e7d32'
        },
        'quality_fill_colors': {
            'Poor': 'rgba(255,75,75,0.25)',
            'Average': 'rgba(255,143,0,0.25)',
            'Good': 'rgba(46,125,50,0.25)'
        }
    }

def create_safe_polar_chart(data_df, title, r_columns, theta_labels, name_column):
    """Create a polar chart with safe color formatting"""
    fig = go.Figure()
    colors = get_safe_colors()
    
    for i, (_, row) in enumerate(data_df.iterrows()):
        color_idx = i % len(colors['solid_colors'])
        
        fig.add_trace(go.Scatterpolar(
            r=[row[col] for col in r_columns],
            theta=theta_labels,
            fill='toself',
            name=str(row[name_column]),
            line=dict(color=colors['solid_colors'][color_idx], width=2),
            fillcolor=colors['fill_colors'][color_idx]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat='.0%'
            )
        ),
        showlegend=True,
        title=title,
        height=400,
        title_font=dict(size=14, color='#722F37')
    )
    
    return fig

# Load all components
loading_result = load_model_components()
components_loaded = loading_result[8]
missing_files = loading_result[9] if len(loading_result) > 9 else []

if components_loaded:
    (model, scaler, label_encoder, feature_names, wine_type_encoder, 
     feature_ranges, dataset_stats, model_performance, _, _) = loading_result
    
    # Professional card style with author info
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #28a745;
        border-radius: 0.375rem;
        padding: 0.75rem 1rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
    ">
        <div style="
            background: rgba(114, 47, 55, 0.1);
            border: 1px solid #722F37;
            border-radius: 6px;
            padding: 0.5rem;
            margin-right: 1rem;
            text-align: center;
            min-width: 120px;
        ">
            <div style="color: #722F37; font-weight: 700; font-size: 0.8rem; margin-bottom: 0.1rem;">
                🎓 Apu Datta
            </div>
            <div style="color: #722F37; font-weight: 500; font-size: 0.7rem; line-height: 1.1;">
                MSBA Student<br>
                Baruch College (CUNY)
            </div>
        </div>
        <div style="color: #155724; font-weight: 600; font-size: 2rem; flex: 1;">
            🍷 Professional Wine Quality Classifier!
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs with enhanced styling
    tab1, tab2 = st.tabs(["🍷 Wine Quality Prediction", "📊 Model Performance Dashboard"])

    with tab1:
        # Enhanced main header
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border: 2px solid #722F37;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0 2rem 0;
            text-align: center;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        ">
            <h3 style="color: #722F37; margin-bottom: 0.8rem; font-weight: 700;">
                🎯 Advanced ML-Powered Wine Analysis
            </h3>
            <p style="font-size: 1.1rem; color: #555; margin-bottom: 0.5rem;">
                Utilizing <strong>{dataset_stats['best_model_type']}</strong> trained on 
                <strong>{dataset_stats['total_samples']:,} real wine samples</strong>
            </p>
            <p style="color: #777; margin: 0; font-size: 0.95rem;">
                Best Model: <strong>{dataset_stats['best_model_name']}</strong> | 
                Accuracy: <strong>{dataset_stats['best_model_accuracy']:.1%}</strong> | 
                Features: <strong>{dataset_stats['feature_count']}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Enhanced sidebar
        with st.sidebar:
            st.markdown('<div style="font-size: 1.1rem; color: #722F37; font-weight: 600; margin: 0.5rem 0; border-left: 3px solid #722F37; padding-left: 0.8rem;">🍷 Wine Properties Configuration</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 0.8rem; color: #666; font-style: italic; margin: 0.2rem 0 0.5rem 0;">Configure wine\'s chemical & physical properties below</div>', unsafe_allow_html=True)

            # Add model components status
            st.markdown("---")
            st.success("✅ All model loaded successfully!")
            
            # Show loaded components
            st.markdown("**📦 Loaded Components:**")
            components_html = '<div style="margin-top: 0.2rem; line-height: 1.1;">'
            if model is not None:
                components_html += '<div style="margin: 0; padding: 0; font-size: 0.85rem;">• 🤖 ML Model</div>'
            if scaler is not None:
                components_html += '<div style="margin: 0; padding: 0; font-size: 0.85rem;">• ⚖️ Scaler</div>'
            if label_encoder is not None:
                components_html += '<div style="margin: 0; padding: 0; font-size: 0.85rem;">• 🏷️ Label Encoder</div>'
            if feature_names is not None:
                components_html += '<div style="margin: 0; padding: 0; font-size: 0.85rem;">• 📋 Feature Names</div>'
            if wine_type_encoder is not None:
                components_html += '<div style="margin: 0; padding: 0; font-size: 0.85rem;">• 💖 Wine Type Encoder</div>'
            if feature_ranges is not None:
                components_html += '<div style="margin: 0; padding: 0; font-size: 0.85rem;">• 📊 Feature Ranges</div>'
            if dataset_stats is not None:
                components_html += '<div style="margin: 0; padding: 0; font-size: 0.85rem;">• 📈 Dataset Stats</div>'
            if model_performance is not None:
                components_html += '<div style="margin: 0; padding: 0; font-size: 0.85rem;">• 🎯 Model Performance</div>'
            components_html += '</div>'
            st.markdown(components_html, unsafe_allow_html=True)
            
            # Dataset and model links
            st.markdown("---")
            st.markdown("**🔗 Resources:**")            
            # Dataset link
            st.markdown("""
            **📊 Dataset Source:**
            - [Wine Quality Dataset - Kaggle](https://www.kaggle.com/datasets/rajyellow46/wine-quality)
            """)
            # Model/Code links
            st.markdown("""
            **🤖 GitHub:**
            - [GitHub Repository](https://github.com/dattaBus-anls/-Professional-Wine-Quality-Classifier-.git)
            """)
            
            # Model details
            st.markdown(f"""
            **📈 Model Details:**
            - **Total Samples:** {dataset_stats['total_samples']:,}
            - **Red Wines:** {dataset_stats['red_wines']:,}
            - **White Wines:** {dataset_stats['white_wines']:,}
            - **Best Algorithm:** {dataset_stats['best_model_name']}
            """)
            
            st.markdown("---")

        # Get wine type options from actual encoder
        wine_type_options = wine_type_encoder.classes_.tolist()
        
        # Create input fields using features from model training
        user_inputs = {}
        
        # Enhanced two-column layout
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown('<div class="section-header">🧪 Acidity & Chemical Balance</div>', unsafe_allow_html=True)
            
            # Enhanced feature labels and help text
            feature_labels = {
                'fixed_acidity': 'Fixed Acidity (g/L)',
                'volatile_acidity': 'Volatile Acidity (g/L)', 
                'citric_acid': 'Citric Acid (g/L)',
                'ph': 'pH Level',
                'chlorides': 'Chlorides (g/L)',
                'sulphates': 'Sulphates (g/L)'
            }
            
            feature_help = {
                'fixed_acidity': 'Tartaric acid - affects wine structure and taste balance',
                'volatile_acidity': 'Acetic acid - high levels create vinegar taste',
                'citric_acid': 'Adds freshness and citrus notes to wine',
                'ph': 'Acidity level (lower values = more acidic)',
                'chlorides': 'Salt content affecting wine taste',
                'sulphates': 'Wine preservative affecting shelf life and quality'
            }
            
            # Dynamically create inputs for acidity-related features
            acidity_features = [f for f in feature_names if f in feature_labels]
            
            for feature in acidity_features:
                user_inputs[feature] = safe_create_slider(
                    feature, feature_ranges, feature_labels, feature_help
                )

        with col2:
            st.markdown('<div class="section-header">🍇 Sugar, Alcohol & Physical Properties</div>', unsafe_allow_html=True)
            
            feature_labels.update({
                'residual_sugar': 'Residual Sugar (g/L)',
                'alcohol': 'Alcohol Content (%)',
                'density': 'Density (g/cm³)',
                'free_sulfur_dioxide': 'Free SO₂ (mg/L)',
                'total_sulfur_dioxide': 'Total SO₂ (mg/L)'
            })
            
            feature_help.update({
                'residual_sugar': 'Remaining sugar after fermentation (affects sweetness)',
                'alcohol': 'Alcohol percentage by volume',
                'density': 'Wine density (affected by sugar and alcohol content)',
                'free_sulfur_dioxide': 'Free sulfur dioxide (prevents oxidation)',
                'total_sulfur_dioxide': 'Total sulfur dioxide content'
            })
            
            # Dynamically create inputs for other features
            other_features = [f for f in feature_names if f in feature_labels and f not in acidity_features]
            
            for feature in other_features:
                user_inputs[feature] = safe_create_slider(
                    feature, feature_ranges, feature_labels, feature_help
                )
            
            # Enhanced wine type selector
            st.markdown("---")
            wine_type = st.selectbox(
                "🍷 Wine Type Selection",
                options=wine_type_options,
                help=f"Select wine type. Available options: {', '.join(wine_type_options)}",
                index=0
            )

        # Encode wine type using the same encoder from training
        wine_type_encoded = wine_type_encoder.transform([wine_type])[0]
        
        # Handle the encoded wine type feature if it exists
        if 'wine_type_encoded' in feature_names:
            user_inputs['wine_type_encoded'] = float(wine_type_encoded)

        # Create final input array in exact order of feature_names
        final_inputs = []
        missing_features = []
        
        for feature_name in feature_names:
            if feature_name in user_inputs:
                final_inputs.append(user_inputs[feature_name])
            else:
                missing_features.append(feature_name)
        
        # Enhanced error handling for missing features
        if missing_features:
            st.error(f"❌ Missing features in model configuration: {missing_features}")
            st.info("Please ensure the model training script generated all required features correctly.")
            st.stop()

        # Enhanced prediction section
        st.markdown("---")
        st.markdown('<div class="section-header">🔮 Wine Quality Prediction</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🔮 Analyze Wine Quality", type="primary", use_container_width=True):
                with st.spinner("🧬 Analyzing wine characteristics using advanced ML algorithms..."):
                    quality, probabilities, success = make_prediction(
                        final_inputs, model, scaler, label_encoder, feature_names
                    )
                    
                    if success:
                        # Enhanced prediction display
                        st.markdown("### 🎯 Prediction Results")
                        
                        # Quality configuration from actual labels
                        quality_config = {
                            "Poor": {"icon": "🔴", "color": "#ff4b4b", "desc": "Not recommended for consumption", "bg": "#ffebee"},
                            "Average": {"icon": "🟡", "color": "#ff8f00", "desc": "Decent table wine", "bg": "#fff8e1"},
                            "Good": {"icon": "🟢", "color": "#2e7d32", "desc": "Premium quality wine", "bg": "#e8f5e8"}
                        }
                        
                        config = quality_config.get(quality, {"icon": "⚪", "color": "#666666", "desc": "Unknown quality", "bg": "#f5f5f5"})
                        
                        st.markdown(f"""
                        <div class="prediction-box" style="background: linear-gradient(135deg, {config['bg']} 0%, #f9f9f9 100%);">
                            <h1 style="color: {config['color']}; margin: 0; font-size: 2.5rem; font-weight: 700;">
                                {config['icon']} {quality} Quality Wine
                            </h1>
                            <p style="margin: 0.8rem 0 0.5rem 0; color: #555; font-size: 1.1rem; font-weight: 500;">
                                {config['desc']}
                            </p>
                            <p style="margin: 0; color: #888; font-size: 0.9rem;">
                                Predicted by <strong>{dataset_stats['best_model_name']}</strong>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Store prediction results in session state for dashboard access
                        st.session_state.last_prediction = {
                            'quality': quality,
                            'probabilities': probabilities,
                            'user_inputs': user_inputs.copy(),
                            'wine_type': wine_type
                        }
                        
                        st.success("✅ Prediction completed! Check the 📊 Model Performance Dashboard tab for detailed analysis.")
                        
                    else:
                        st.error("❌ Prediction failed. Please check inputs and try again.")
                        st.info("Ensure all model files are properly generated from the training script.")

        # Professional footer
        st.markdown("---")
        st.markdown(f"""
        <div style="
            text-align: center; 
            color: #666; 
            margin-top: 2rem;
            padding: 1.5rem;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px;
            border: 1px solid #dee2e6;
        ">
            <h4 style="color: #722F37; margin-bottom: 1rem;">🍷 Professional Wine Quality Classifier</h4>
            <p style="margin: 0.5rem 0;">
                <strong>Trained on {dataset_stats['total_samples']:,} Real Wine Samples</strong> | 
                Powered by {dataset_stats['best_model_type']} Machine Learning
            </p>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">
                Technologies: Scikit-learn • Streamlit • Plotly • Python
            </p>
            <p style="margin: 0; font-style: italic; font-size: 0.85rem;">
                All predictions based on actual training results - No hardcoded values
            </p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        # ENHANCED MODEL DASHBOARD TAB
        st.markdown("""
        <h3 style="
            font-size: 1.5rem; 
            color: #722F37; 
            font-weight: 600;
            text-align: center; 
            margin-top: 1rem;
            margin-bottom: 2rem;
        ">
        📊 Comprehensive Model Performance Dashboard
        </h3>
        """, unsafe_allow_html=True)

        if not (model_performance and dataset_stats):
            st.error("❌ Model performance data not available")
            st.info("Please ensure model training completed successfully and all .pkl files are present.")
            st.stop()

        # MOVED SECTIONS FROM TAB 1 TO TAB 2

        # 1. MODEL CONFIDENCE SECTION (if prediction exists)
        if hasattr(st.session_state, 'last_prediction') and st.session_state.last_prediction:
            st.markdown("---")
            prediction_data = st.session_state.last_prediction
            
            if hasattr(model, 'predict_proba'):
                display_model_confidence(prediction_data['probabilities'], label_encoder)
            else:
                st.info("Model confidence analysis requires a model with probability support.")
        
        # 2. WINE ANALYSIS SUMMARY (if prediction exists)
        if hasattr(st.session_state, 'last_prediction') and st.session_state.last_prediction:
            st.markdown("---")
            prediction_data = st.session_state.last_prediction
            display_wine_metrics_professional(prediction_data['user_inputs'], prediction_data['wine_type'])
        
        # 3. MODEL INFORMATION & PERFORMANCE SECTION
        st.markdown("---")
        display_model_info_performance(dataset_stats, model_performance)
        
        # 4. TOP PERFORMING MODELS SECTION
        st.markdown("---")
        display_top_performing_models(model_performance)

        # Enhanced progress tracking for remaining dashboard sections
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            status_text.text("🔄 Loading additional dashboard components...")
            progress_bar.progress(20)
            
            # ENHANCED DASHBOARD OVERVIEW
            st.markdown("---")
            st.markdown('<div class="section-header">📈 Training Performance Overview</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                best_model_short = dataset_stats['best_model_name'][:15] + "..." if len(dataset_stats['best_model_name']) > 15 else dataset_stats['best_model_name']
                st.metric(
                    "🎯 Best Model", 
                    best_model_short,
                    f"{dataset_stats['best_model_accuracy']:.1%} Accuracy",
                    help=f"Full name: {dataset_stats['best_model_name']}"
                )
            
            with col2:
                st.metric(
                    "📊 Training Data", 
                    f"{dataset_stats['total_samples']:,}",
                    "Wine Samples",
                    help="Total number of wine samples used for training"
                )
            
            with col3:
                st.metric(
                    "🔬 Features", 
                    dataset_stats['feature_count'],
                    "Wine Properties",
                    help="Number of chemical and physical properties analyzed"
                )
            
            with col4:
                quality_counts = dataset_stats['quality_distribution']
                total_classes = len(quality_counts)
                st.metric(
                    "📈 Quality Classes", 
                    total_classes,
                    "Categories",
                    help="Poor, Average, Good quality classifications"
                )

            progress_bar.progress(40)
            status_text.text("📊 Generating algorithm comparison...")
            
            # ENHANCED MODEL PERFORMANCE COMPARISON
            try:
                st.markdown("---")
                st.markdown('<div class="section-header">🏆 Algorithm Performance Comparison</div>', unsafe_allow_html=True)
                
                test_accs = model_performance['test_accuracies']
                
                # Limit to top 12 models for better visualization
                if len(test_accs) > 12:
                    sorted_models = sorted(test_accs.items(), key=lambda x: x[1], reverse=True)[:12]
                    test_accs = dict(sorted_models)
                    st.info("📊 Showing top 12 models for optimal visualization")
                
                performance_data = []
                for model_name, accuracy in test_accs.items():
                    # Shorten long model names for better display
                    display_name = model_name[:30] + "..." if len(model_name) > 30 else model_name
                    performance_data.append({
                        'Model': display_name,
                        'Full_Name': model_name,
                        'Test_Accuracy': accuracy,
                        'Model_Type': 'Best Model' if model_name == dataset_stats['best_model_name'] else 'Other Models'
                    })
                
                perf_df = pd.DataFrame(performance_data)
                perf_df = perf_df.sort_values('Test_Accuracy', ascending=True)
                
                # Enhanced horizontal bar chart
                fig_performance = px.bar(
                    perf_df, 
                    x='Test_Accuracy', 
                    y='Model',
                    color='Model_Type',
                    color_discrete_map={'Best Model': '#722F37', 'Other Models': '#cccccc'},
                    title="🎯 Model Test Accuracy Comparison",
                    labels={'Test_Accuracy': 'Test Accuracy (%)', 'Model': 'Machine Learning Models'},
                    hover_data={'Full_Name': True, 'Test_Accuracy': ':.1%'}
                )
                
                fig_performance.update_layout(
                    height=600,
                    showlegend=True,
                    xaxis=dict(
                        tickformat='.1%',
                        title_font=dict(size=14, color='#722F37'),
                        range=[0, 1]
                    ),
                    yaxis=dict(
                        title_font=dict(size=14, color='#722F37')
                    ),
                    title=dict(
                        font=dict(size=16, color='#722F37'),
                        x=0.5
                    ),
                    plot_bgcolor='rgba(248,249,250,0.8)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                fig_performance.update_traces(
                    texttemplate='%{x:.1%}', 
                    textposition='outside',
                    textfont=dict(size=10, color='black')
                )
                
                st.plotly_chart(fig_performance, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error in Model Performance Comparison: {str(e)}")
                st.info("Skipping Model Performance Comparison section. Other sections should still work.")
            
            progress_bar.progress(60)
            status_text.text("🔄 Processing cross-validation results...")
            
            # ENHANCED CROSS VALIDATION RESULTS
            try:
                st.markdown("---")
                st.markdown('<div class="section-header">🔄 Cross-Validation Analysis (5-Fold)</div>', unsafe_allow_html=True)
                
                cv_scores = model_performance['cv_scores']
                cv_data = []
                
                for model_name, scores in cv_scores.items():
                    cv_data.append({
                        'Model': model_name[:25] + "..." if len(model_name) > 25 else model_name,
                        'Full_Name': model_name,
                        'Mean_CV_Score': np.mean(scores),
                        'Std_CV_Score': np.std(scores),
                        'Min_Score': np.min(scores),
                        'Max_Score': np.max(scores),
                        'Scores': scores
                    })
                
                cv_df = pd.DataFrame(cv_data)
                cv_df = cv_df.sort_values('Mean_CV_Score', ascending=False)
                
                col1, col2 = st.columns([1.2, 1.8])
                
                with col1:
                    st.markdown('<div class="subsection-header">📈 Top Cross-Validation Performers</div>', unsafe_allow_html=True)
                    top_cv = cv_df.head(8).copy()  # Show top 8
                    top_cv['CV Score'] = top_cv['Mean_CV_Score'].apply(lambda x: f"{x:.2%}")
                    top_cv['Std Dev'] = top_cv['Std_CV_Score'].apply(lambda x: f"±{x:.3f}")
                    top_cv['Range'] = top_cv.apply(lambda row: f"{row['Min_Score']:.3f} - {row['Max_Score']:.3f}", axis=1)
                    
                    display_df = top_cv[['Model', 'CV Score', 'Std Dev', 'Range']].copy()
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Model": st.column_config.TextColumn("🤖 Model", width="medium"),
                            "CV Score": st.column_config.TextColumn("📊 Score", width="small"),
                            "Std Dev": st.column_config.TextColumn("📈 Std Dev", width="small"),
                            "Range": st.column_config.TextColumn("📊 Range", width="medium")
                        }
                    )
                
                with col2:
                    st.markdown('<div class="subsection-header">📊 CV Score Distribution Analysis</div>', unsafe_allow_html=True)
                    
                    # Enhanced box plot for CV scores
                    top_5_models = cv_df.head(5)
                    
                    fig_cv = go.Figure()
                    
                    colors = ['#722F37', '#8b4513', '#a0522d', '#cd853f', '#daa520']
                    
                    for i, (_, row) in enumerate(top_5_models.iterrows()):
                        fig_cv.add_trace(go.Box(
                            y=row['Scores'],
                            name=row['Model'],
                            boxpoints='all',
                            jitter=0.3,
                            pointpos=-1.8,
                            marker=dict(color=colors[i % len(colors)]),
                            line=dict(color=colors[i % len(colors)])
                        ))
                    
                    fig_cv.update_layout(
                        title="🎯 Cross-Validation Score Distribution (Top 5 Models)",
                        yaxis_title="Cross-Validation Score",
                        xaxis_title="Models",
                        height=400,
                        showlegend=False,
                        plot_bgcolor='rgba(248,249,250,0.8)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        title_font=dict(size=14, color='#722F37')
                    )
                    
                    st.plotly_chart(fig_cv, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error in Cross-Validation Analysis: {str(e)}")
                st.info("Skipping Cross-Validation Analysis section. Other sections should still work.")
            
            progress_bar.progress(80)
            status_text.text("📊 Building detailed metrics analysis...")
            
            # ENHANCED DETAILED PERFORMANCE METRICS
            try:
                st.markdown("---")
                st.markdown('<div class="section-header">📊 Detailed Performance Metrics Analysis</div>', unsafe_allow_html=True)
                
                performance_metrics = model_performance['performance_metrics']
                
                # Create comprehensive metrics table
                metrics_data = []
                for model_name, metrics in performance_metrics.items():
                    overall_score = (metrics['accuracy'] + metrics['precision'] + metrics['recall'] + metrics['f1_score']) / 4
                    metrics_data.append({
                        'Model': model_name[:25] + "..." if len(model_name) > 25 else model_name,
                        'Full_Name': model_name,
                        'Accuracy': metrics['accuracy'],
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1_Score': metrics['f1_score'],
                        'Overall_Score': overall_score
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df = metrics_df.sort_values('Overall_Score', ascending=False)
                
                # Format for display
                display_metrics = metrics_df.copy()
                for col in ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'Overall_Score']:
                    display_metrics[col] = display_metrics[col].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(
                    display_metrics[['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'Overall_Score']],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Model": st.column_config.TextColumn("🤖 Model Algorithm", width="large"),
                        "Accuracy": st.column_config.TextColumn("🎯 Accuracy", width="small"),
                        "Precision": st.column_config.TextColumn("📊 Precision", width="small"),
                        "Recall": st.column_config.TextColumn("📈 Recall", width="small"),
                        "F1_Score": st.column_config.TextColumn("⚖️ F1-Score", width="small"),
                        "Overall_Score": st.column_config.TextColumn("🏆 Overall", width="small")
                    }
                )
                
                # Enhanced Performance Metrics Visualization
                st.markdown('<div class="subsection-header">📊 Top Models Performance Visualization</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Radar chart for top 3 models - SAFE VERSION
                    top_3_models = metrics_df.head(3)
                    
                    fig_radar = create_safe_polar_chart(
                        top_3_models, 
                        "🎯 Performance Metrics Comparison (Top 3)",
                        ['Accuracy', 'Precision', 'Recall', 'F1_Score'],
                        ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        'Model'
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                with col2:
                    # Enhanced bar chart for metrics comparison
                    top_5_models = metrics_df.head(5)
                    
                    metrics_viz_data = []
                    for _, row in top_5_models.iterrows():
                        for metric in ['Accuracy', 'Precision', 'Recall', 'F1_Score']:
                            metrics_viz_data.append({
                                'Model': row['Model'],
                                'Metric': metric,
                                'Score': row[metric]
                            })
                    
                    metrics_viz_df = pd.DataFrame(metrics_viz_data)
                    
                    fig_metrics = px.bar(
                        metrics_viz_df,
                        x='Metric',
                        y='Score',
                        color='Model',
                        barmode='group',
                        title="📊 Detailed Metrics Comparison (Top 5)",
                        labels={'Score': 'Performance Score', 'Metric': 'Performance Metrics'}
                    )
                    
                    fig_metrics.update_layout(
                        height=400,
                        yaxis=dict(tickformat='.0%'),
                        title_font=dict(size=14, color='#722F37'),
                        plot_bgcolor='rgba(248,249,250,0.8)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig_metrics, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error in Detailed Performance Metrics: {str(e)}")
                st.info("Skipping Detailed Performance Metrics section. Other sections should still work.")
            
            progress_bar.progress(100)
            status_text.text("✅ Dashboard loaded successfully!")
            
            # Clean up progress indicators
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            # Enhanced professional footer for dashboard
            st.markdown("---")
            st.markdown("""
            <div style="
                text-align: center; 
                color: #666; 
                margin-top: 2rem;
                padding: 2rem;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 15px;
                border: 1px solid #dee2e6;
                box-shadow: 0 4px 16px rgba(0,0,0,0.08);
            ">
                <h3 style="color: #722F37; margin-bottom: 1rem;">📊 Professional ML Model Dashboard</h3>
                <p style="margin: 0.5rem 0; font-size: 1.1rem;">
                    <strong>100% Data-Driven Analytics</strong> • All metrics sourced from actual training results
                </p>
                <p style="margin: 0.5rem 0;">
                    🔬 No hardcoded values • 📊 Real performance metrics • 🎯 Production-ready insights
                </p>
                <p style="margin: 0; font-style: italic; font-size: 0.9rem;">
                    Dashboard automatically synced with Wine_data_processing_model_training_testing.py results
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        except Exception as dashboard_error:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Dashboard error: {dashboard_error}")
            st.info("Try refreshing the page or ensure all model files are properly generated from the training script.")

else:
    # Enhanced error handling section
    st.error("❌ Cannot load model components")
    
    if missing_files:
        st.markdown("### 🔧 Missing Required Files:")
        for file in missing_files:
            st.markdown(f"- ❌ `{file}`")
    
    st.markdown("""
    ### 🚀 Setup Instructions:
    
    **Step 1:** Run the model training script:
    ```bash
    python Wine_data_processing_model_training_testing.py
    ```
    
    **Step 2:** Verify these files are created:
    """)
    
    required_files = [
        'wine_model.pkl',
        'wine_scaler.pkl',  
        'wine_label_encoder.pkl',
        'wine_feature_names.pkl',
        'wine_type_encoder.pkl',
        'feature_ranges.pkl',
        'dataset_stats.pkl',
        'model_performance.pkl'
    ]
    
    for file in required_files:
        status = "✅" if os.path.exists(file) else "❌"
        size_info = ""
        if os.path.exists(file):
            try:
                size = os.path.getsize(file)
                size_info = f" ({size:,} bytes)"
            except:
                size_info = " (size unknown)"
        
        st.markdown(f"- {status} `{file}`{size_info}")
    
    st.markdown("""
    **Step 3:** Run this Streamlit app:
    ```bash
    streamlit run wine_streamlit_app.py
    ```
    
    ### 📋 Troubleshooting:
    - Ensure have all required dependencies installed
    - Check that the training script completed without errors
    - Verify all .pkl files were generated successfully
    - Make sure both scripts are in the same directory
    """)
    
    st.info("💡 **Tip:** The training script must complete successfully before running this Streamlit app. All model components are dynamically loaded from the training results.")