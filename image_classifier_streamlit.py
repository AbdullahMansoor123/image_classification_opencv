import cv2
import numpy as np
import streamlit as st

# step1 load model
st.title('Image Classifier using Opencv')
with open('model/classification_classes_ILSVRC2012.txt', 'r') as f:
    class_names = f.read().split('\n')

    classes = [i.split(',')[0] for i in class_names]

    model = cv2.dnn.readNet(
        model='model/DenseNet_121.caffemodel',
        config='model/DenseNet_121.prototxt',
        framework='Caffe'
    )


types = ['png', 'jpg', 'jpeg']
upload_file = st.file_uploader('Choose an Image File', type=types)
if upload_file is not None:
    raw_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    st.image(img, channels='BGR')
else:
    img_path = 'shark.jpg'
    img = cv2.imread(img_path)
    st.image(img,channels='BGR')
# print(img.shape)

    blob_img = cv2.dnn.blobFromImage(img, scalefactor=0.01, size=(224, 224), mean=(104, 117, 123))
    model.setInput(blob_img)
    output = model.forward()
    # print(output.shape)
    final_output = output[0]
    final_output = final_output.reshape(1000, 1)
    output_class_idx = np.argmax(final_output)
    output_class = classes[output_class_idx]
    probs = np.exp(final_output) / sum(np.exp(final_output))
    output_prob = np.max(probs)
    output_stats = f'Class: {output_class}\n Probability: {output_prob:.3f}'
    st.write(output_stats)

