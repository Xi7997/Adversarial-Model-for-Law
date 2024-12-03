import numpy as np
import tensorflow as tf
from openai import OpenAI
import matplotlib.pyplot as plt
import os
import json
import csv
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
client = OpenAI()
class GenerativeContractiveModel:
    def __init__(self):
        original_data = []
        original_label = []
        generative_data = []
        generative_label = []
        prediction = []
        Log = []

    def load_data(self, original_path, generative_path):
        with open(original_path, 'r') as file:
            lines = json.load(file)
            for line in lines:
                self.original_data.append(line['data'])
                self.original_label.append(line['label'])
        with open(generative_path, 'r') as file:
            lines = json.load(file)
            for line in lines:
                self.generative_data.append(line['data'])
                self.generative_label.append(line['label'])

    def train(self):
        tokenizer = AutoTokenizer.from_pretrained("law-ai/InCaseLawBERT")
        original_data = tokenizer(self.original_data, padding=True, truncation=True, return_tensors='pt')
        generative_data = tokenizer(self.generative_data, padding=True, truncation=True, return_tensors='pt')
        original_dataset = TensorDataset(original_data['input_ids'], original_data['attention_mask'], torch.tensor(self.original_label))
        generative_dataset = TensorDataset(generative_data['input_ids'], generative_data['attention_mask'], torch.tensor(self.generative_label))
        original_loader = DataLoader(original_dataset, batch_size=32, shuffle=True)
        generative_loader = DataLoader(generative_dataset, batch_size=32, shuffle=True)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(30522, 768, input_length=512),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(original_loader, validation_data=generative_loader, epochs=10)
        self.prediction = model.predict(generative_loader)
        self.Log = model.history

    def save(self, path):
        with open(path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['original_data', 'original_label', 'generative_data', 'generative_label', 'prediction', 'Log'])
            writer.writerow([self.original_data, self.original_label, self.generative_data, self.generative_label, self.prediction, self.Log])


    def load(self, path):
        with open(path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            self.original_data, self.original_label, self.generative_data, self.generative_label, self.prediction, self.Log = next(reader)


if __name__ == '__main__':
    model = GenerativeContractiveModel()
    model.load_data('original.json', 'generative.json')
    model.train()
    model.save('model.csv')
    model.load('model.csv')
    print(model.prediction)
    print(model.Log)


