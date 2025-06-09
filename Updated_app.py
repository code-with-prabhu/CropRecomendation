import streamlit as st
import pickle
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

# Load the saved model and LabelEncoder
with open('model_new.pkl', 'rb') as model_file:
    model, enc = pickle.load(model_file)

# Title of the app
st.set_page_config(page_title="Crop Recommender System", layout="wide")
st.title("🌾 Crop Recommender System")

# Add a background image (optional)
st.markdown("""
<style>
body {
    background-image: url("https://www.example.com/your-background-image.jpg");
    background-size: cover;
}
</style>
""", unsafe_allow_html=True)

# Sidebar for user input features
st.sidebar.header("🌱 Input Soil Parameters")


def get_user_input():
    # Define input fields with descriptions
    st.sidebar.subheader("Soil Nutrients")
    N = st.sidebar.number_input("Nitrogen (N) content (kg/ha)", min_value=0.0, step=0.1)
    P = st.sidebar.number_input("Phosphorus (P) content (kg/ha)", min_value=0.0, step=0.1)
    K = st.sidebar.number_input("Potassium (K) content (kg/ha)", min_value=0.0, step=0.1)

    st.sidebar.subheader("Soil Properties")
    ph = st.sidebar.number_input("pH level", min_value=0.0, max_value=14.0, step=0.1)
    EC = st.sidebar.number_input("Electrical Conductivity (EC) (dS/m)", min_value=0.0, step=0.1)

    st.sidebar.subheader("Trace Elements")
    S = st.sidebar.number_input("Sulfur (S) content (ppm)", min_value=0.0, step=0.1)
    Cu = st.sidebar.number_input("Copper (Cu) content (ppm)", min_value=0.0, step=0.1)
    Fe = st.sidebar.number_input("Iron (Fe) content (ppm)", min_value=0.0, step=0.1)
    Mn = st.sidebar.number_input("Manganese (Mn) content (ppm)", min_value=0.0, step=0.1)
    Zn = st.sidebar.number_input("Zinc (Zn) content (ppm)", min_value=0.0, step=0.1)
    B = st.sidebar.number_input("Boron (B) content (ppm)", min_value=0.0, step=0.1)

    # Combine all inputs into a DataFrame
    features = {
        'N': N,
        'P': P,
        'K': K,
        'ph': ph,
        'EC': EC,
        'S': S,
        'Cu': Cu,
        'Fe': Fe,
        'Mn': Mn,
        'Zn': Zn,
        'B': B
    }
    user_input = pd.DataFrame([features])
    return user_input


# Get user input
user_input = get_user_input()

# Display the input data for confirmation
st.subheader("🔍 User Input Features")
st.write(user_input)

# Initialize a variable to store prediction result
prediction = None
predicted_label = None

# Add a button to trigger prediction
if st.button('🔮 Recommend'):
    # Make predictions
    prediction = model.predict(user_input)

    # Decode the numerical prediction to the actual label
    predicted_label = enc.inverse_transform(prediction)

    # Show the prediction result
    st.subheader("🌟 Predicted Crop")
    st.write(f"**Predicted Crop**: {predicted_label[0]}")

    # Show the prediction probabilities (optional)
    if hasattr(model, 'predict_proba'):
        prediction_proba = model.predict_proba(user_input)
        st.subheader("📊 Prediction Probability")
        st.write(prediction_proba)

# Add a button to generate and download the PDF report
if st.button('📄 Generate Report') and predicted_label is not None:
    # Create a BytesIO buffer for the PDF
    buffer = BytesIO()

    # Create a PDF
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Add content to the PDF
    c.drawString(50, height - 50, "Crop Recommender System Report")
    c.drawString(50, height - 100, "User Input Data:")

    # Add user input data to the PDF
    y_position = height - 150
    for index, value in user_input.iloc[0].items():
        c.drawString(50, y_position, f"{index}: {value}")
        y_position -= 20

    c.drawString(50, y_position - 20, f"Predicted Crop: {predicted_label[0]}")

    # Finalize the PDF
    c.save()

    # Move to the beginning of the StringIO buffer
    buffer.seek(0)

    # Provide download link
    st.download_button(
        label="Download Report",
        data=buffer,
        file_name="crop_recommender_report.pdf",
        mime="application/pdf"
    )
