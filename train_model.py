import pandas as pd
from datasets import Datasetgit commit -m "first commit"
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Загрузка данных
train_df = pd.read_csv("data/train.csv")

dataset = Dataset.from_pandas(train_df)

# Маппинг меток
label2id = {"negative": 0, "positive": 1}
id2label = {0: "negative", 1: "positive"}

dataset = dataset.map(lambda x: {"label": label2id[x["label"]]})

# Токенизатор
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)

# Модель
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# Параметры обучения
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Обучение
trainer.train()

# Сохранение модели
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")

print("Fine-tuning завершён")
