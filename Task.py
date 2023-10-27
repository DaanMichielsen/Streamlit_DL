import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.tree import export_graphviz
from PIL import Image
import pickle
import base64
import zipfile
import math
import gdown
import tensorflow as tf
from keras.models import load_model
from keras import optimizers
import tensorflow.keras.utils as np_utils
from keras.utils import image_dataset_from_directory
from sklearn.metrics import confusion_matrix

url = 'https://drive.google.com/drive/folders/1-KtOahPkEev5_KwNbGx8YLpjO3JRNaNU?usp=sharing'
output = 'weights_folder.zip'
gdown.download(url, output, quiet=False)

# Extract the contents of the zip file
#with zipfile.ZipFile(output, 'r') as zip_ref:
#    zip_ref.extractall('saved_models')


def embed_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    st.markdown(pdf_display, unsafe_allow_html=True)

def load_weights(epochs):
    epochs_str = "{:02d}".format(epochs)
    model.load_weights(f'./weights_{epochs_str}.tf')

import os
import zipfile

def unzip_test_data():
    # Replace './test_data/val.zip' with the path to your zip file
    zip_path = './val.zip'

    # Replace './test_data/' with the path to your directory
    extract_path = './test_data/'

    # Check if the directory is empty
    if not os.listdir(extract_path):
        # Extract the contents of the zip file to the specified directory
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    else:
            pass
# pdf_path = "ML_Benchmark_Daan_Michielsen.pdf"
# with open(pdf_path, "rb") as f:
#         pdf_bytes = f.read()

# st.set_page_config(layout='wide')
unzip_test_data()
st.title("Image classification using a CNN :robot_face::brain::frame_with_picture:")
st.caption('''
Created by [Daan Michielsen](https://github.com/DaanMichielsen)
           ''')

st.header("Categories", divider='violet')
st.markdown('''
I selected the following categories for my images to scrape using Selenium in Python:
        \n- Banana
        \n- Bottle
        \n- Capybara
        \n- French Bulldog
        \n- Pizza
            \n Each of the categories contains 200 images.
            \nI decided the keep all the scraped images because I was quite satisfied with the quality of them.
''')
st.image(['./selenium.png'], width=100)

st.header("Training my own CNN on the images", divider='violet')
st.markdown("I created a **convolutional neural network** which consists of **multitple convolutional layers** with numerous neurons which **look for features** in the images **by applying filters**. For **improving** the model I also use **augmentation** by **using zoom, tilt, flipping and cropping to create \"new\" images from existing images** so our model has more images to work with.")
st.markdown("For the training I split the images into :blue[training(640)], :orange[validation(160)] and test set(200).")

epochs = 15
# Display the slider and update the test size
epochs = st.slider('### Select amount of epochs:', min_value=1, max_value=30, step=1, value=epochs)

# Create a sequential model with a list of layers
model = load_model('./models/my_model.h5')

load_weights(epochs)

# Compile and train your model as usual
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #compiling our model to use ADAM as optimizer, categorical cross entropy calculation for finding our loss

with open('./history/history.pkl', 'rb') as f:
    history = pickle.load(f)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

st.subheader("Plotting the loss and accuracy curves for the model", divider="violet")

# Plot the loss curves
ax1.plot(history['loss'][:epochs])
ax1.plot(history['val_loss'][:epochs])
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend(['Train', 'Validation'], loc='upper right')

# Plot the accuracy curves
ax2.plot(history['accuracy'][:epochs])
ax2.plot(history['val_accuracy'][:epochs])
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend(['Train', 'Validation'], loc='lower right')

# Display the figure
st.pyplot(fig)

# Create two columns to display the values for the selected epoch
col1, col2 = st.columns(2)
col1.subheader('Loss for epoch {}'.format(epochs))
col1.write('Train: :blue[{:.4f}]'.format(history['loss'][epochs-1]))
col1.write('Validation: :orange[{:.4f}]'.format(history['val_loss'][epochs-1]))

col2.subheader('Accuracy for epoch {}'.format(epochs))
col2.write('Train: :blue[{:.4f}]'.format(history['accuracy'][epochs-1]))
col2.write('Validation: :orange[{:.4f}]'.format(history['val_accuracy'][epochs-1]))

# Replace '/content/drive/MyDrive/datasets/val' with the path to your test directory
test_dir = './test_data'

# Define the batch size and image size
batch_size = 64
image_size = (256, 256)
classes = ['Banana','Bottle','Capybara','FR bulldog','Pizza']
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=test_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )
# Create a function to extract the test dataset and make predictions
def predict():
    msg = st.toast("Starting predictions...", icon="👨‍🔬")
    st.subheader("Prediction results", divider='violet')
    tab_classes = ['Banana:banana:','Bottle:baby_bottle:','Capybara:amphora:','FR Bulldog:dog:','Pizza:pizza:']
    banana, bottle, capybara, french_bulldog, pizza = st.tabs(tab_classes)
    # Make predictions on the test dataset using the loaded model
    y_pred = model.predict(test_ds)
    msg.toast("Predictions Completed", icon="✔️")
    msg.toast("plotting results...", icon="📈")
    # Get the images and labels from the testing dataset
    x_test = []
    y_test = []

    for images, labels in test_ds:
        x_test.append(images.numpy())
        y_test.append(labels.numpy())

    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)

    # Normalize the image data
    x_test_norm = x_test / 255.0

    # Print the images with their labels
    plt.figure(figsize=(10,10))
    plt.subplots_adjust(hspace=0.5)
    fig, axs = plt.subplots(5, 8, figsize=(10, 10))
    with banana:
        num_correct = 0
        num_wrong = 0
        for i, ax in enumerate(axs.flat):
            ax.imshow(x_test_norm[i], cmap='gray')
            actual_label = classes[np.argmax(y_test[i])]
            predicted_label = classes[np.argmax(y_pred[i])]
            probability = y_pred[i][np.argmax(y_pred[i])]
            if actual_label == predicted_label:
                label_color = "green"
                num_correct += 1
            else:
                label_color = "red"
                num_wrong += 1
            ax.set_xlabel(f"{actual_label} (a)\n{predicted_label} (p)\nProb: {math.floor(probability* 1000) / 1000}", color=label_color)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        right, wrong = st.columns(2)        
        with right:
            st.markdown(f'**Right predictions:** :green[{num_correct}]')
        with wrong:
            st.markdown(f'**Wrong predictions:** :red[{num_wrong}]')
        

        # Add padding between each row of subplots
        plt.subplots_adjust(hspace=0.5)
        st.pyplot(fig)
    with bottle:
        num_correct = 0
        num_wrong = 0
        for i, ax in enumerate(axs.flat):
            ax.imshow(x_test_norm[i+40], cmap='gray')
            actual_label = classes[np.argmax(y_test[i+40])]
            predicted_label = classes[np.argmax(y_pred[i+40])]
            probability = y_pred[i+40][np.argmax(y_pred[i+40])]
            if actual_label == predicted_label:
                label_color = "green"
                num_correct += 1
            else:
                label_color = "red"
                num_wrong += 1
            ax.set_xlabel(f"{actual_label} (a)\n{predicted_label} (p)\nProb: {math.floor(probability* 1000) / 1000}", color=label_color)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        right, wrong = st.columns(2)        
        with right:
            st.markdown(f'**Right predictions:** :green[{num_correct}]')
        with wrong:
            st.markdown(f'**Wrong predictions:** :red[{num_wrong}]')

        # Add padding between each row of subplots
        plt.subplots_adjust(hspace=0.5)
        st.pyplot(fig)
    with capybara:
        num_correct = 0
        num_wrong = 0
        for i, ax in enumerate(axs.flat):
            ax.imshow(x_test_norm[i+80], cmap='gray')
            actual_label = classes[np.argmax(y_test[i+80])]
            predicted_label = classes[np.argmax(y_pred[i+80])]
            probability = y_pred[i+80][np.argmax(y_pred[i+80])]
            if actual_label == predicted_label:
                label_color = "green"
                num_correct += 1
            else:
                label_color = "red"
                num_wrong += 1
            ax.set_xlabel(f"{actual_label} (a)\n{predicted_label} (p)\nProb: {math.floor(probability* 1000) / 1000}", color=label_color)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        right, wrong = st.columns(2)        
        with right:
            st.markdown(f'**Right predictions:** :green[{num_correct}]')
        with wrong:
            st.markdown(f'**Wrong predictions:** :red[{num_wrong}]')

        # Add padding between each row of subplots
        plt.subplots_adjust(hspace=0.5)
        st.pyplot(fig)
    with french_bulldog:
        num_correct = 0
        num_wrong = 0
        for i, ax in enumerate(axs.flat):
            ax.imshow(x_test_norm[i+120], cmap='gray')
            actual_label = classes[np.argmax(y_test[i+120])]
            predicted_label = classes[np.argmax(y_pred[i+120])]
            probability = y_pred[i+120][np.argmax(y_pred[i+120])]
            if actual_label == predicted_label:
                label_color = "green"
                num_correct += 1
            else:
                label_color = "red"
                num_wrong += 1
            ax.set_xlabel(f"{actual_label} (a)\n{predicted_label} (p)\nProb: {math.floor(probability* 1000) / 1000}", color=label_color)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        right, wrong = st.columns(2)        
        with right:
            st.markdown(f'**Right predictions:** :green[{num_correct}]')
        with wrong:
            st.markdown(f'**Wrong predictions:** :red[{num_wrong}]')

        # Add padding between each row of subplots
        plt.subplots_adjust(hspace=0.5)
        st.pyplot(fig)
    with pizza:
        num_correct = 0
        num_wrong = 0
        for i, ax in enumerate(axs.flat):
            ax.imshow(x_test_norm[i+160], cmap='gray')
            actual_label = classes[np.argmax(y_test[i+160])]
            predicted_label = classes[np.argmax(y_pred[i+160])]
            probability = y_pred[i+160][np.argmax(y_pred[i+160])]
            if actual_label == predicted_label:
                label_color = "green"
                num_correct += 1
            else:
                label_color = "red"
                num_wrong += 1
            ax.set_xlabel(f"{actual_label} (a)\n{predicted_label} (p)\nProb: {math.floor(probability* 1000) / 1000}", color=label_color)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        right, wrong = st.columns(2)        
        with right:
            st.markdown(f'**Right predictions:** :green[{num_correct}]')
        with wrong:
            st.markdown(f'**Wrong predictions:** :red[{num_wrong}]')

        # Add padding between each row of subplots
        plt.subplots_adjust(hspace=0.5)
        st.pyplot(fig)
    num_correct = 0
    num_wrong = 0
    for i in range(len(y_pred)):
        actual_label = classes[np.argmax(y_test[i])]
        predicted_label = classes[np.argmax(y_pred[i])]
        if actual_label == predicted_label:
            num_correct += 1
        else:
            num_wrong += 1
    st.subheader("Overall results")
    st.markdown(f'**Right predictions:** :green[{num_correct}]')
    st.markdown(f'**Wrong predictions:** :red[{num_wrong}]')
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    pie , matrix = st.columns(2)
    with pie:
        # Create a pie chart
        fig, ax = plt.subplots()
        ax.pie([num_correct, num_wrong], labels=['Correct','Wrong'], autopct='%1.1f%%', colors=['green', 'red'])

        # Display the pie chart using st.pyplot()
        st.pyplot(fig)
    with matrix:
        cm = confusion_matrix(y_test_classes, y_pred_classes)

        # Create a DataFrame object for your confusion matrix
        confusion_matrix_df = pd.DataFrame(cm, index=[c for c in classes], columns=[c for c in classes])

        # Create a heatmap of your confusion matrix using seaborn
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix_df, annot=True, fmt='d', cmap='Blues', annot_kws={'size': 16})

        # Set the axis labels and title
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')

        # Display the plot in Streamlit
        st.pyplot(fig)
    st.subheader("Most confused predictions:confused:")
    st.markdown("These results represent the predictions the model was really really sure it was right about, but ended up being wrong.")
    # Find the indices of the images where your model made a wrong prediction
    wrong_indices = np.nonzero(y_pred_classes != y_test_classes)[0]

    # Find the indices of the images where your model was most confused
    most_confused_indices = wrong_indices[np.argsort(np.max(y_pred[wrong_indices], axis=1))[-10:]]
    # Your code for creating the plot
    label_color = "red"
    fig, axs = plt.subplots(2, 5, figsize=(10, 5))
    for index, i in enumerate(most_confused_indices):
        row = index // 5
        col = index % 5
        axs[row, col].imshow(x_test_norm[i], cmap='gray')
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])
        axs[row, col].grid(False)
        actual_label = classes[np.argmax(y_test[i])]
        predicted_label = classes[np.argmax(y_pred[i])]
        probability = y_pred[i][np.argmax(y_pred[i])]
        label = f"{actual_label} (a)\n{predicted_label} (p)\nProbability:\n {math.floor(probability* 100000) / 100000}"
        axs[row, col].set_xlabel(label, color=label_color)

    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.5)

    # Display the plot in Streamlit
    st.pyplot(fig)
    st.markdown("These are the **10 worst** predictions the model made. It seems like predicting capybara's is not it's best quality.:man-shrugging:")

    msg.toast("Plotting completed", icon="✔️")

# Create a button to trigger predictions on the test dataset
if st.button(f'Predict for {epochs} epochs', use_container_width=True):
    with st.spinner('Running prediction...'):
        predict()


# st.download_button(
#     label="Download PDF",
#     data=pdf_bytes,
#     file_name="DL_Daan_Michielsen.pdf",
#     mime="application/pdf"
# )
# with st.expander("View PDF"):
#     embed_pdf(pdf_path=pdf_path)
