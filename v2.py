# Must be the first Streamlit command
import streamlit as st
st.set_page_config(page_title="Crop Recommender System", layout="wide")


import pickle
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from io import BytesIO
from datetime import datetime

# Initialize session state
if 'predicted_label' not in st.session_state:
    st.session_state.predicted_label = None
if 'prediction_proba' not in st.session_state:
    st.session_state.prediction_proba = None
if 'user_input' not in st.session_state:
    st.session_state.user_input = None


# Load the saved model and LabelEncoder
@st.cache_resource
def load_model():
    with open('model_new.pkl', 'rb') as model_file:
        return pickle.load(model_file)


model, enc = load_model()


def setup_page():
    """Configure the Streamlit page settings"""
    st.title("🌾 Crop Recommender System")

    # Add custom CSS
    st.markdown("""
    <style>
    body {
        background-image: url("https://miro.medium.com/v2/resize:fit:800/1*KdG6q6RR5EEowxkxTxrWbA.jpeg");
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)


def get_user_input():
    """Collect and validate user input parameters"""
    st.sidebar.header("🌱 Input Soil Parameters")

    input_sections = {
        "Soil Nutrients": {
            "N": ("Nitrogen (N) content (kg/ha)", 0.0, None, 0.1),
            "P": ("Phosphorus (P) content (kg/ha)", 0.0, None, 0.1),
            "K": ("Potassium (K) content (kg/ha)", 0.0, None, 0.1)
        },
        "Soil Properties": {
            "ph": ("pH level", 0.0, 14.0, 0.1),
            "EC": ("Electrical Conductivity (EC) (dS/m)", 0.0, None, 0.1)
        },
        "Trace Elements": {
            "S": ("Sulfur (S) content (ppm)", 0.0, None, 0.1),
            "Cu": ("Copper (Cu) content (ppm)", 0.0, None, 0.1),
            "Fe": ("Iron (Fe) content (ppm)", 0.0, None, 0.1),
            "Mn": ("Manganese (Mn) content (ppm)", 0.0, None, 0.1),
            "Zn": ("Zinc (Zn) content (ppm)", 0.0, None, 0.1),
            "B": ("Boron (B) content (ppm)", 0.0, None, 0.1)
        }
    }

    features = {}
    for section, params in input_sections.items():
        st.sidebar.subheader(section)
        for key, (label, min_val, max_val, step) in params.items():
            features[key] = st.sidebar.number_input(
                label,
                min_value=min_val,
                max_value=max_val,
                step=step
            )

    return pd.DataFrame([features])


def generate_pdf_report(user_input, predicted_label, prediction_proba=None):
    """Generate a detailed PDF report with the analysis results"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )

    # Prepare the story (content) for the PDF
    story = []
    styles = getSampleStyleSheet()

    # Add custom style for headers
    styles.add(ParagraphStyle(
        name='CustomHeader',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30
    ))

    # Add title and date
    story.append(Paragraph("Crop Recommender System Report", styles['CustomHeader']))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))

    # Add soil analysis data
    story.append(Paragraph("Soil Analysis Results", styles['Heading2']))
    story.append(Spacer(1, 12))

    # Create table data
    table_data = [['Parameter', 'Value', 'Unit']]
    units = {
        'N': 'kg/ha', 'P': 'kg/ha', 'K': 'kg/ha',
        'ph': '', 'EC': 'dS/m',
        'S': 'ppm', 'Cu': 'ppm', 'Fe': 'ppm',
        'Mn': 'ppm', 'Zn': 'ppm', 'B': 'ppm'
    }

    for param, value in user_input.iloc[0].items():
        table_data.append([param, f"{value:.2f}", units.get(param, '')])

    # Create and style the table
    table = Table(table_data, colWidths=[200, 100, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    story.append(table)
    story.append(Spacer(1, 20))

    # Add prediction results
    story.append(Paragraph("Prediction Results", styles['Heading2']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Recommended Crop: {predicted_label}", styles['Normal']))

    # Add probability distribution if available
    if prediction_proba is not None:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Probability Distribution:", styles['Normal']))
        proba_data = [['Crop', 'Probability']]
        for crop, prob in zip(enc.classes_, prediction_proba[0]):
            proba_data.append([crop, f"{prob:.2%}"])

        proba_table = Table(proba_data, colWidths=[200, 100])
        proba_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(proba_table)

    # Build the PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


def main():
    """Main application logic"""
    setup_page()

    # Get user input
    st.session_state.user_input = get_user_input()

    # Display the input data
    st.subheader("🔍 User Input Features")
    st.write(st.session_state.user_input)
    # Show generate report button
    col1, col2= st.columns([1,1])

    with col1:
        # Make prediction
        if st.button('🔮 Recommend'):
            prediction = model.predict(st.session_state.user_input)
            st.session_state.predicted_label = enc.inverse_transform(prediction)[0]

            if hasattr(model, 'predict_proba'):
                st.session_state.prediction_proba = model.predict_proba(st.session_state.user_input)

            st.subheader("🌟 Predicted Crop")
            st.write(f"**Predicted Crop**: {st.session_state.predicted_label}")

            if st.session_state.prediction_proba is not None:
                st.subheader("📊 Prediction Probability")
                st.write(st.session_state.prediction_proba)


    with col1:
        if st.button('📄 Generate Report'):
            if st.session_state.predicted_label is None:
                st.error("Please click 'Recommend' first to get the prediction!")
            else:
                pdf_buffer = generate_pdf_report(
                    st.session_state.user_input,
                    st.session_state.predicted_label,
                    st.session_state.prediction_proba
                )

                st.download_button(
                    label="📥 Download Report",
                    data=pdf_buffer,
                    file_name=f"crop_recommendation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )


if __name__ == "__main__":
    main()