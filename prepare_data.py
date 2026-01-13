import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Загрузка данных
df = pd.read_csv("data/raw_data.csv")

# Удаление дубликатов
df = df.drop_duplicates()

# Функция очистки текста
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

# Сохранение очищенных данных
df.to_csv("data/cleaned_data.csv", index=False)

# Разделение на train / test
train_df, test_df = train_test_split(
    df,
    test_size=0.25,
    random_state=42,
    stratify=df["label"]
)

train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("Данные подготовлены успешно")
