import os
import base64
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# 设置OpenAI的API密钥
api_key = 'O'
# 定义图片文件夹路径
image_folder = r'D:\project_driver_detection\car_act\data_part_80'

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to send image to OpenAI and get the response
def get_image_classification(image_path):
    base64_image = encode_image(image_path)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt = """
    You are an well-trained advanced model discriminator proficient in analyzing driver behavior from an in-car perspective.

    Please classify the driver's behavior into one of the following categories by only returning the category code (c0 to c7) without any additional text or explanation:
    c0: safe driving
    c1: texting
    c2: talking on the phone
    c3: operating the radio
    c4: drinking
    c5: reaching behind
    c6: hair and makeup
    c7: talking to passenger

    You should focus more on the actions of the person sitting in the driver's seat and carefully distinguish between the states provided (c0 to c7). Here are some annotations to help you:
    - safe driving (c0): The driver is looking ahead with both hands on the steering wheel.
    - texting (c1): The driver is looking at their phone, not close to the ear.
    - talking on the phone (c2): The driver is holding the phone close to their ear or face.
    - operating the radio (c3):The driver’s right hand reaches towards the central console to operate the controls, with only one hand remaining on the steering wheel.
    - drinking (c4): The driver is holding a water bottle or cup with one hand.
    - reaching behind (c5): The driver’s right hand is stretched towards the rear seats of the vehicle and is not visible in the image.
    - hair and makeup (c6): The driver is not holding a phone and is using a hand to groom their hair or apply makeup, possibly in front of a mirror.
    - talking to passenger (c7): The driver is engaging in conversation with the passenger.

    Please carefully observe the actions of the driver in the driver's seat. Try to improve the accuracy of your response through careful observation of the driver!

    Additional guidelines to help with classification:
    - If the driver has both hands on the steering wheel, it can only be c0 or c7 (but c7 will clearly show the driver talking to someone, and c7 might not always have both hands on the steering wheel).
    - If the driver is holding a phone with one hand, it can only be c1 or c2.
    - If the driver is holding a cup or bottle, it can only be c4.
    - If the driver’s right hand is extended forward and not fully visible in the image, it can only be c3.
    - If the driver’s right hand is extended backward and not fully visible in the image, it can only be c5.
    - If the driver’s hand is near their head or face without holding anything, it can only be c6.
    - The driver must have both hands on the steering wheel and be looking ahead for safe driving.
    
    Respond with only the category code (e.g., c0, c1, c2, etc.).
    """

    payload = {
        #"model": "gpt-4o-mini",
        "model": "gpt-4o",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

# Define classification labels
labels = {
    'c0': 'safe driving',
    'c1': 'texting',
    'c2': 'talking on the phone',
    'c3': 'operating the radio',
    'c4': 'drinking',
    'c5': 'reaching behind',
    'c6': 'hair and makeup',
    'c7': 'talking to passenger'
}

# Initialize lists to store actual and predicted labels
actual_labels = []
predicted_labels = []

# Iterate over all files in the folder
for image_file in os.listdir(image_folder):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        actual_category_code = image_file.split('_')[0]
        actual_labels.append(actual_category_code)

        image_path = os.path.join(image_folder, image_file)

        # Retry logic for classification
        success = False
        while not success:
            try:
                classification = get_image_classification(image_path)
                category_code = classification['choices'][0]['message']['content'].strip()
                category_label = labels.get(category_code, 'Unknown category')
                predicted_labels.append(category_code)
                print(f"Image: {image_file}, Classification: {category_code} - {category_label}")
                success = True
            except KeyError:
                print(f"Retrying for image: {image_file}")

# Calculate accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)
print(f"Overall Accuracy: {accuracy}")

# Compute confusion matrix
conf_matrix = confusion_matrix(actual_labels, predicted_labels, labels=[f'c{i}' for i in range(8)])

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=[f'c{i}' for i in range(8)], yticklabels=[f'c{i}' for i in range(8)])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()