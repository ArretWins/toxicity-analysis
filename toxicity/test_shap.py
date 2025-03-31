import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd

# Загрузка модели и токенизатора
model_name = "minuva/MiniLMv2-toxic-jigsaw"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Функция для предсказания вероятностей
def predict_proba(texts):
    inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs).logits
        probabilities = torch.sigmoid(outputs).numpy()
    return probabilities

# Тестовый текст
test_text = "I hate niggers"

# Проверка predict_proba
probs = predict_proba([test_text])
print("Predicted probabilities:", probs)

# Генерация объяснения SHAP
def generate_shap_explanation(text):
    try:
        # Создаем SHAP Explainer
        explainer = shap.Explainer(lambda x: predict_proba(x), shap.maskers.Text(tokenizer))
        shap_values = explainer([text])

        # Преобразуем токены и атрибуции
        encoded = tokenizer(text, return_offsets_mapping=True)
        words = tokenizer.convert_ids_to_tokens(encoded["input_ids"])[1:-1]
        offsets = encoded["offset_mapping"][1:-1]
        importances = shap_values.values[0].sum(axis=1)

        # Группируем подтокены в слова
        word_importance = {}
        current_word = ""
        current_importance = 0
        previous_end = -1

        for (token, importance, (start, end)) in zip(words, importances, offsets):
            if start != previous_end and current_word:
                word_importance[current_word] = current_importance
                current_word = ""
                current_importance = 0
            current_word += token.replace("##", "")
            current_importance += importance
            previous_end = end

        if current_word:
            word_importance[current_word] = current_importance

        # Выводим важность слов
        print("SHAP features:", word_importance)

        # Создаем график
        df = pd.DataFrame(list(word_importance.items()), columns=["Word", "Importance"])
        plt.figure(figsize=(10, 5))
        plt.barh(df["Word"], df["Importance"], color="skyblue")
        plt.xlabel("SHAP Importance")
        plt.ylabel("Word")
        plt.title("Word Importance in Toxicity Classification")
        plt.gca().invert_yaxis()

        # Сохраняем график в base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png).decode("utf-8")
        return graphic

    except Exception as e:
        print(f"Error generating SHAP explanation: {str(e)}")
        return None

# Генерируем объяснение SHAP для тестового текста
shap_graph = generate_shap_explanation(test_text)
if shap_graph:
    print("SHAP graph generated successfully.")
else:
    print("Failed to generate SHAP graph.")