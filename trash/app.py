from flask import Flask, render_template,  request
import pandas as pd
import pickle
from typing import Tuple
import tensorflow as tf
import numpy as np
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import img_to_array
import pickle
from flask import Flask, render_template, url_for, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img
from keras.preprocessing import image
import sqlite3
import shutil

from googletrans import Translator

# Create translator instance
translator = Translator()


import google.generativeai as genai
genai.configure(api_key='AIzaSyBOzq7o6-tmDja-xH71m1JvzC3b1l7pvug')
gemini_model = genai.GenerativeModel('gemini-1.5-pro')
chat = gemini_model.start_chat(history=[])


model3 = pickle.load(open("0-3.sav", "rb"))
model11 = pickle.load(open("4-11.sav", "rb"))
MODEL_PATH = 'image_model.tflite'
modelmri=load_model('monument_classifier.h5')
with open('class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
chat_history = []
chat_history_kn = []
chat_history_hi = []
app = Flask(__name__)
##############################################################################################
three_q=["Does your child look at you when you call his/her name?",
"How easy is it for you to get eye contact with your child",
"Does your child point to indicate that s/he want something",
"Does your child point to share intrest with you?(eg. pointing at intresting sights)",
"Does your child pretend ?(eg. care for dolls,talk on a toy phone)",
"Does your child follow where you're looking?",
"If you or someone else in the family is visibly upset ,does your child show signs of wanting to comfort them?(eg. stroking hair,hugging them)",
"Would you describe your child's first word",
"Does your child use simple gestures?(eg. wave goodbye)",
"Does your child stare at nothing with no apparent purpose",
"age",
"gender",
"etnicity",
"Was the child infected with jaundice?",
"Is ASD trait found in any family member?"]

eleven_q=["S/he often notices small sounds when others do not?",
"S/he usually concentrates more on the whole picture, rather than the small details?",
"In a social group, s/he can easily keep track of several different people’s conversations?",
"S/he finds it easy to go back and forth between different activities?",
"S/he doesn’t know how to keep a conversation going with his/her peers?",
"S/he is good at social chit-chat?",
"When s/he is read a story, s/he finds it difficult to work out the character’s intentions or feelings?",
"When s/he was in preschool, s/he used to enjoy playing games involving pretending with other children?",
"S/he finds it easy to work out what someone is thinking or feeling just by looking at their face?",
"S/he finds it hard to make new friends?",
"age",
"Gender",
"etnicity",
"Was the child infected with jaundice?",
"Is ASD trait found in any family member?"]
##################################################################################################


def get_interpreter(model_path: str) -> Tuple:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return interpreter, input_details, output_details

def predict(image_path: str) -> int:
    interpreter, input_details, output_details = get_interpreter(MODEL_PATH)
    input_shape = input_details[0]['shape']
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.resize(img, (input_shape[2], input_shape[2]))
    img = tf.expand_dims(img, axis=0)
    resized_img = tf.cast(img, dtype=tf.uint8)
    
    interpreter.set_tensor(input_details[0]['index'], resized_img)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    return np.argmax(results, axis=0),results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyse', methods=['GET', 'POST'])
def analyse():
    if request.method == 'POST':
        user_input = request.form['query']
        if user_input == "hi":
            chat_history.append([user_input, "Hi , How are you this is Mental Health related chatbot assistance"])
        # Get response from Gemini AI model
        else:
            gemini_response = chat.send_message(user_input+"for this  query give response  and i know providing medical advice online is risky so dont show excess contents (note: provide in html format content)")
            data = gemini_response.text
            data = data.replace("```html", "")
            data = data.replace("```", "")
            chat_history.append([user_input, data])
            
        return render_template('chatbot.html', chat_history=chat_history)
    return render_template('chatbot.html')

@app.route('/analyseh', methods=['GET', 'POST'])
def analyseh():
    if request.method == 'POST':
        user_input = request.form['query']
        if user_input.lower() in ["hi", "hello", "नमस्ते"]:
            response_text = "नमस्ते, आज मैं आपकी कैसे सहायता कर सकता हूँ?"

            
        else:
            from_lang = 'hi'
            to_lang = 'en'
            text_to_translate = translator.translate(user_input, src=from_lang, dest=to_lang)
            text = text_to_translate.text  
            
            gemini_response = chat.send_message(text+"for this  query give response  and i know providing medical advice online is risky so dont show excess contents (note: provide in html format content)")
            data = gemini_response.text
            data = data.replace("```html", "")
            response = data.replace("```", "")
            
            print(f"Bot: {response}")

            from_lang = 'en'
            to_lang = 'hi'
            text_to_translate = translator.translate(response, src=from_lang, dest=to_lang)
            text = text_to_translate.text

            # TTS(text, 'hi')
            # chat_history_hi.append([user_input, data])
            chat_history_hi.append([user_input, text])
            
            
        return render_template('chatboth.html', chat_history_hi=chat_history_hi)
    return render_template('chatboth.html')

@app.route('/analysek', methods=['GET', 'POST'])
def analysek():
    if request.method == 'POST':
        user_input = request.form['query']
        if user_input.lower() in ["hi", "hello", "ನಮಸ್ತೆ", "ನಮಸ್ಕಾರ"]:
            response_text = "ಹಾಯ್, ನಾನು ಇಂದು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?"
        else:
            from_lang = 'kn'
            to_lang = 'en'
            text_to_translate = translator.translate(user_input, src=from_lang, dest=to_lang)
            text = text_to_translate.text  


            gemini_response = chat.send_message(text+"for this  query give response  and i know providing medical advice online is risky so dont show excess contents (note: provide in html format content)")
            data = gemini_response.text
            data = data.replace("```html", "")
            response     = data.replace("```", "")
            

            # response = chatbot_response(text)
            print(f"Bot: {response}")

            from_lang = 'en'
            to_lang = 'kn'
            text_to_translate = translator.translate(response, src=from_lang, dest=to_lang)
            text = text_to_translate.text

            

            chat_history_kn.append([user_input, text])
            return render_template('chatbotk.html',chat_history_kn=chat_history_kn)
        
            
            
        return render_template('chatbotk.html', chat_history_kn=chat_history_kn)
    return render_template('chatbotk.html')



@app.route('/Three_year', methods=['GET', 'POST'])
def Three_year():
    if request.method == 'POST':
        df = request.form
        data = []
        data.append(int(df['A1']))
        data.append(int(df['A2']))
        data.append(int(df['A3']))
        data.append(int(df['A4']))
        data.append(int(df['A5']))
        data.append(int(df['A6']))
        data.append(int(df['A7']))
        data.append(int(df['A8']))
        data.append(int(df['A9']))
        data.append(int(df['A10']))

        if int(df['age']) < 12:
            data.append(0)
        else:
            data.append(1)
        
        data.append(int(df['gender']))

        if df['etnicity'] == 'middle eastern':
            data.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])	
        if df['etnicity'] == 'White European':	
            data.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'Hispanic':
            data.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'black':
            data.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])	
        if df['etnicity'] == 'asian':	
            data.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'south asian':
            data.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])	
        if df['etnicity'] == 'Native Indian':
            data.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        if df['etnicity'] == 'Others':	
            data.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        if df['etnicity'] == 'Latino':
            data.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])	
        if df['etnicity'] == 'mixed':
            data.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        if df['etnicity'] == 'Pacifica':
            data.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        data.append(int(df['Jaundice']))
        data.append(int(df['ASD']))

        name = df['name']
        email = df['email']
        print(name)
        print(email)

        Index = model3.predict([data])
        if Index == 0:
            prediction = 'Non-autistic'
            # return render_template('index.html', name=name, email=email,  prediction=prediction)
        else:
            prediction = 'Autistic'
        chk=[int(df['A1']),int(df['A2']),int(df['A3']),int(df['A4']),int(df['A5']),
                int(df['A6']),int(df['A7']),int(df['A8']),int(df['A9']),int(df['A10']),
                int(df['age']),int(df['gender']),df['etnicity'],df['Jaundice'],df['ASD']]
        prmpt=f"user got prediction like {prediction} where the inputs considered  {three_q} and the input provided is {chk} so explain why {prediction} occured ,write in short (give response in html tags to make it display directly)"
        gemini_response = chat.send_message(prmpt)
        recommendatoin = gemini_response.text
        recommendatoin = recommendatoin.replace("```html", "")
        recommendatoin = recommendatoin.replace("```", "")
        print("\n\n\n diagnosis tip \n\n\n")
        return render_template('index.html', name=name, email=email,  prediction=prediction,recommendatoin=recommendatoin)
            
        

        
    return render_template('index.html')

@app.route('/Eleven_year', methods=['GET', 'POST'])
def Eleven_year():
    if request.method == 'POST':
        df = request.form
        data = []
        data.append(int(df['A1']))
        data.append(int(df['A2']))
        data.append(int(df['A3']))
        data.append(int(df['A4']))
        data.append(int(df['A5']))
        data.append(int(df['A6']))
        data.append(int(df['A7']))
        data.append(int(df['A8']))
        data.append(int(df['A9']))
        data.append(int(df['A10']))

        if int(df['age']) < 12:
            data.append(0)
        else:
            data.append(1)
        
        data.append(int(df['gender']))

        if df['etnicity'] == 'Others':
            data.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])	
        if df['etnicity'] == 'Middle Eastern':	
            data.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'Hispanic':
            data.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'White-European':
            data.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])	
        if df['etnicity'] == 'Black':	
            data.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        if df['etnicity'] == 'South Asian':
            data.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])	
        if df['etnicity'] == 'Asian':
            data.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        if df['etnicity'] == 'Pasifika':	
            data.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        if df['etnicity'] == 'Turkish':
            data.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        if df['etnicity'] == 'Latino':
            data.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        data.append(int(df['Jaundice']))
        data.append(int(df['ASD']))

        name = df['name']
        email = df['email']
        print(name)
        print(email)

        Index = model11.predict([data])
        if Index == 0:
            prediction = 'Non-autistic'
        else:
            prediction = 'Autistic'
        print(prediction)
        chk=[int(df['A1']),int(df['A2']),int(df['A3']),int(df['A4']),int(df['A5']),
                int(df['A6']),int(df['A7']),int(df['A8']),int(df['A9']),int(df['A10']),
                int(df['age']),int(df['gender']),df['etnicity'],df['Jaundice'],df['ASD']]
        prmpt=f"user got prediction like {prediction} where the inputs considered  {three_q} and the input provided is {chk} so explain why {prediction} occured ,write in short (give response in html tags to make it display directly)"
        gemini_response = chat.send_message(prmpt)
        recommendatoin = gemini_response.text
        recommendatoin = recommendatoin.replace("```html", "")
        recommendatoin = recommendatoin.replace("```", "")
        print("\n\n\n diagnosis tip \n\n\n")

        return render_template('index.html', name=name, email=email,  prediction=prediction,recommendatoin=recommendatoin)
    return render_template('index.html')

@app.route('/Image', methods=['GET', 'POST'])
def Image():
    if request.method == 'POST':
        name = request.form['name']
        filename = request.form['filename']
        email = request.form['email']
        path = 'static/test/'+filename
        Index,results = predict(path)
        print(results[0])
        print(results[1])

        print(name)
        if Index == 0:
            prediction = 'Non-autistic'
        else:
            prediction = 'Autistic'
        prmpt=f"Chance of autistic is {results[1]} and Chance of Non-Autistic is {results[0]} for the patient face features through cnn so explain why {prediction} diagnosised ,write in short (give response in html tags to make it display directly)"
        gemini_response = chat.send_message(prmpt)
        recommendatoin = gemini_response.text
        recommendatoin = recommendatoin.replace("```html", "")
        recommendatoin = recommendatoin.replace("```", "")
        print(prediction)
        

        return render_template('index.html', name=name, email=email, prediction=prediction, img='http://127.0.0.1:5000/'+path,recommendatoin=recommendatoin)
    return render_template('index.html')

@app.route('/mri_image', methods=['GET', 'POST'])
def mri_image():
    if request.method == 'POST':
        name = request.form['name']
        filename = request.form['filename']
        email = request.form['email']
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        
        

        shutil.copy("static/testmri/"+fileName, dst)
        image = cv2.imread("satic/testmri/"+fileName)

        # model=load_model('monument_classifier.h5')
        path='static/images/'+fileName


        # # Load the class names
        # with open('class_names.pkl', 'rb') as f:
        #     class_names = pickle.load(f)
        dec=""
        dec1=""
        # Function to preprocess the input image
        def preprocess_input_image(path):
            img = load_img(path, target_size=(150,150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize the image
            return img_array

        # Function to make predictions on a single image
        def predict_single_image(path):
            input_image = preprocess_input_image(path)
            prediction = modelmri.predict(input_image)
            print(prediction)
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_names[predicted_class_index]
            confidence = prediction[0][predicted_class_index]

            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
                
            return predicted_class, confidence ,prediction[0]

        predicted_class, confidence,pred = predict_single_image(path)
        #predicted_class, confidence = predict_single_image(path, model, class_names)
        print(predicted_class, confidence)
        if predicted_class == 'AUTISTIC':
            str_label = "AUTISTIC"

           
        elif predicted_class == 'NORMAL':
            str_label = "NORMAL"
        print(pred)
        prmpt=f"Chance of autistic is {pred[0]} and Chance of Non-Autistic is {pred[1]} for the patient Mri features through cnn so explain why {str_label} diagnosised ,write in short (give response in html tags to make it display directly)"
        gemini_response = chat.send_message(prmpt)
        recommendatoin = gemini_response.text
        recommendatoin = recommendatoin.replace("```html", "")
        recommendatoin = recommendatoin.replace("```", "")

            
        accuracy = f"The predicted image is {str_label} with a confidence of {confidence:.2%}"

        return render_template('index.html', name=name, email=email, prediction=str_label, img='http://127.0.0.1:5000/'+path,recommendatoin=recommendatoin)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)