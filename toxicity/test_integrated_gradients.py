import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients
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

# Генерация объяснения Integrated Gradients
def generate_integrated_gradients_explanation(text):
    try:
        # Токенизация текста
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Получаем эмбеддинги из модели
        embeddings = model.get_input_embeddings()(input_ids)

        # Создаем forward-функцию для Captum
        def forward_func(input_embeds):
            return model(inputs_embeds=input_embeds, attention_mask=attention_mask).logits

        # Создаем объяснитель Integrated Gradients
        ig = IntegratedGradients(forward_func)

        # Вычисляем атрибуции
        attributions, _ = ig.attribute(embeddings, target=0, return_convergence_delta=True)
        attributions = attributions.sum(dim=-1).squeeze(0)  # Суммируем по последней оси
        attributions = attributions / torch.norm(attributions)  # Нормализуем значения

        # Преобразуем токены и атрибуции
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        # Убираем служебные токены ([CLS], [SEP], [PAD])
        filtered_tokens = []
        filtered_attributions = []
        for token, attr in zip(tokens, attributions.tolist()):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue  # Пропускаем служебные токены
            filtered_tokens.append(token)
            filtered_attributions.append(attr)

        # Объединяем подтокены в слова
        merged_tokens, merged_attributions = merge_subtokens(filtered_tokens, filtered_attributions)

        # Выводим важность слов
        print("Integrated Gradients features:", dict(zip(merged_tokens, merged_attributions)))

        # Создаем график
        plt.figure(figsize=(10, 6))
        colors = ['red' if attr > 0 else 'blue' for attr in merged_attributions]
        bars = plt.barh(merged_tokens, merged_attributions, color=colors, edgecolor='black')

        for bar, attr in zip(bars, merged_attributions):
            plt.text(
                bar.get_width() + 0.01 if attr > 0 else bar.get_width() - 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{attr:.2f}",
                va='center',
                ha='left' if attr > 0 else 'right',
                fontsize=9
            )

        plt.title("Integrated Gradients: Word Importance", fontsize=14, fontweight='bold')
        plt.xlabel("Attribution Score", fontsize=12)
        plt.ylabel("Words", fontsize=12)

        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png).decode("utf-8")
        return graphic

    except Exception as e:
        print(f"Error generating Integrated Gradients explanation: {str(e)}")
        return None

# Функция для объединения подтокенов в слова
def merge_subtokens(tokens, attributions):
    merged_tokens = []
    merged_attributions = []
    current_token = ""
    current_attr = 0.0

    for token, attr in zip(tokens, attributions):
        if token.startswith("##"):
            current_token += token[2:]
            current_attr += attr
        else:
            if current_token:
                merged_tokens.append(current_token)
                merged_attributions.append(current_attr)
            current_token = token
            current_attr = attr

    if current_token:
        merged_tokens.append(current_token)
        merged_attributions.append(current_attr)

    return merged_tokens, merged_attributions

# Генерируем объяснение Integrated Gradients для тестового текста
ig_graph = generate_integrated_gradients_explanation(test_text)
if ig_graph:
    print("Integrated Gradients graph generated successfully.")
else:
    print("Failed to generate Integrated Gradients graph.")