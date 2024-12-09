import streamlit as st

# Set page configuration
st.set_page_config(page_title="Health Insurance Prediction", page_icon="🏠", layout="wide")

# Title and description
st.title("Health Insurance Prediction")
st.write("""
    This is a simple web app to predict the health insurance charges based on the data provided by the user.
    The data is used to train a machine learning model that predicts the charges based on the input data.
    The model is trained using the [Medical Cost Personal Datasets](https://www.kaggle.com/mirichoi0218/insurance) from Kaggle.
""")

# Display an image (replace 'path_to_image' with the actual path or URL of your image)
st.image('https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.TUdRFk3_qfy7lAQRO7SLkQHaHa%26pid%3DApi&f=1&ipt=e6058fe6893d943099fb82458fa993f1ea284b73ae0f6d4b1b26a0b3db965c37&ipo=images', caption="Health Insurance Prediction Model")

# Additional description
st.write("""
    To get started, there are some visualizations of the data used for training the model.
    You can also input your own data to get a prediction of the charges.
""")
