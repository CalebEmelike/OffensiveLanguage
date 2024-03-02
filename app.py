from simpletransformers.classification import ClassificationModel
import os
import streamlit as st



# Set tokenizers parallelism environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"


args = {
    "use_multiprocessing": False,
    "use_multiprocessing_for_evaluation": False,
    "process_count": 1
}

model_path = '/Users/CalebE/Downloads/data/content/outputs'

# Replace 'outputs' with your directory path if different
model = ClassificationModel('roberta', model_path, use_cuda=False)

prediction_label = {0: 'This Sentence is not offensive', 1: 'This Sentence is a Targeted Insult', 2: 'This Sentence is not a Targeted Insult, but is offensive'}

st.title("Offensive Language Detection")

text = st.text_area("Enter Text", value='Type Here')

if st.button('Predict'):
    # Make prediction
    prediction, raw_outputs = model.predict([text])
    # Map prediction to label
    prediction = prediction_label[prediction[0]]
    # Display prediction
    st.write(f'Prediction: {prediction}')