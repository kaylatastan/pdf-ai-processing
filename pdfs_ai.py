import requests
import tempfile
import pdfplumber
import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from concurrent.futures import ThreadPoolExecutor


toc_keywords = ['İçindekiler', 'Contents', 'Table of Contents']
tot_keywords = ['Tablolar', 'Tables', 'List of Tables']
tof_keywords = ['Şekiller', 'Figures', 'List of Figures']
references_keywords = ['Kaynakça', 'References', 'Bibliography']
indexes_keywords = ['Dizin', 'Indexes', 'Index', 'Dizinler']

def detect_special_page(text):
    if any(re.search(keyword, text, re.IGNORECASE) for keyword in toc_keywords):
        return 'TOC'
    elif any(re.search(keyword, text, re.IGNORECASE) for keyword in tot_keywords):
        return 'TOT'
    elif any(re.search(keyword, text, re.IGNORECASE) for keyword in tof_keywords):
        return 'TOF'
    elif any(re.search(keyword, text, re.IGNORECASE) for keyword in references_keywords):
        return 'References'
    elif any(re.search(keyword, text, re.IGNORECASE) for keyword in indexes_keywords):
        return 'Indexes'
    else:
        return 'Normal Page' 

def fetch_pdf_data(url):
    response = requests.get(url)
    pdf_data = []

    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(response.content)
        pdf_path = tf.name

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                page_type = detect_special_page(text)
                pdf_data.append({
                    'page_number': i + 1,
                    'text': text,
                    'page_type': page_type
                })
    
    os.remove(pdf_path)
    return pdf_data

pdf_urls = [
    "https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/pdfreference1.7old.pdf",
    "https://www.newinchess.com/media/wysiwyg/product_pdf/8234.pdf"
]

all_pages_data = []
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(fetch_pdf_data, pdf_urls))
    for result in results:
        all_pages_data.extend(result)

df = pd.DataFrame(all_pages_data)

df['label'] = df['page_type'].apply(lambda x: 0 if x == 'Normal Page' else 1)

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])
y_train = train_df["label"]
y_test = test_df["label"]


lr_model = LogisticRegression(max_iter=100)
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)
print("Logistic Regression Model")
print(classification_report(y_test, lr_predictions))


nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


adam_optimizer = Adam(learning_rate=0.001)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

nn_model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

nn_loss, nn_accuracy = nn_model.evaluate(X_test, y_test)
print(f"Neural Network Test Accuracy: {nn_accuracy}")
