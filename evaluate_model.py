import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Загрузка модели
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model")
model = AutoModelForSequenceClassification.from_pretrained("fine_tuned_model")

# Тестовые данные
test_df = pd.read_csv("data/test.csv")

correct = 0

for _, row in test_df.iterrows():
    inputs = tokenizer(row["text"], return_tensors="pt")
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

    predicted_label = model.config.id2label[prediction]

    if predicted_label == row["label"]:
        correct += 1

accuracy = correct / len(test_df)

print(f"Accuracy: {accuracy:.2f}")
