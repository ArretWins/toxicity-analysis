import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from io import BytesIO
import base64

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
test_text = "I hate you but I love peace"

# Проверка predict_proba
probs = predict_proba([test_text])
print("Predicted probabilities:", probs)

# Генерация объяснения LIME
explainer = LimeTextExplainer(class_names=["toxic", "non-toxic"])
explanation = explainer.explain_instance(
    test_text,
    lambda x: predict_proba(x),
    num_features=10,
    labels=[0]  # Анализируем только первый класс (токсичность)
)

# Вывод важности слов
lime_features = explanation.as_list(label=0)
print("LIME features:", lime_features)

# График LIME
def generate_lime_explanation(text):
    try:
        explainer = LimeTextExplainer(class_names=["toxic", "non-toxic"])
        explanation = explainer.explain_instance(
            text,
            lambda x: predict_proba(x),
            num_features=10,
            labels=[0]
        )
        lime_features = explanation.as_list(label=0)
        tokens, importances = zip(*lime_features)

        # Создаем график
        plt.figure(figsize=(10, 6))
        colors = ['red' if imp > 0 else 'blue' for imp in importances]
        bars = plt.barh(tokens, importances, color=colors, edgecolor='black')

        # Добавляем числовые значения над столбцами
        for bar, imp in zip(bars, importances):
            plt.text(
                bar.get_width() + 0.01 if imp > 0 else bar.get_width() - 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{imp:.2f}",
                va='center',
                ha='left' if imp > 0 else 'right',
                fontsize=9
            )

        plt.title("LIME: Word Importance", fontsize=14, fontweight='bold')
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Words", fontsize=12)

        # Сохраняем график в base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png).decode("utf-8")
        return graphic

    except Exception as e:
        print(f"Error generating LIME explanation: {str(e)}")
        return None

# Генерируем график для тестового текста
lime_graph = generate_lime_explanation(test_text)
if lime_graph:
    print("LIME graph generated successfully.")
else:
    print("Failed to generate LIME graph.")