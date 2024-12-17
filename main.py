

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from ultralytics import YOLO
import numpy as np
from PIL import Image
from openai import OpenAI


client = OpenAI(api_key ="open api key")


def classify_image_pretrained(image_path):
    model = keras.applications.ResNet50(weights='imagenet')
    img = Image.open(image_path).resize((224, 224))
    x = np.array(img)
    x = keras.applications.resnet50.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    decoded_predictions = keras.applications.resnet50.decode_predictions(predictions, top=3)[0]
    scene_classes = [f"{class_name} ({confidence:.2f})" for _, class_name, confidence in decoded_predictions]
    return scene_classes

def detect_objects(image_path, model_path="yolo11n.pt"):
    model = YOLO(model_path)
    results = model(image_path)
    detected_objects = []
    for result in results[0].boxes.data:
        class_id = int(result[5])
        label = model.names[class_id]
        detected_objects.append(label)
    return detected_objects



def generate_textual_description(scene_classes, detected_objects, context=None):
    prompt = f"""
    Given the following image analysis results:
    - Scene detected: {', '.join(scene_classes)}
    - Objects detected: {', '.join(detected_objects)}

    Generate a detailed, context-rich caption for the image.
    """
    if context:
        prompt += f" Also, include the context of '{context}' in the description."

    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates detailed image captions."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return chat_completion.choices[0].message.content

def analyze_image(image_path, context=None):
    print("Analyzing Image...")
    scene_classes = classify_image_pretrained(image_path)
    print("Scene Classes:", scene_classes)
    detected_objects = detect_objects(image_path)
    print("Detected Objects:", detected_objects)
    textual_description = generate_textual_description(scene_classes, detected_objects, context)
    print("\nGenerated Caption:")
    print(textual_description)

if __name__ == "__main__":
    while True:
        image_path = input("Enter the path to the image: ")
        if os.path.exists(image_path):
            user_keyword = input("Enter a keyword for context (e.g., 'summer'): ")
            analyze_image(image_path, context=user_keyword)
            break
        else:
            print("The file does not exist. Please enter a valid image path.")
