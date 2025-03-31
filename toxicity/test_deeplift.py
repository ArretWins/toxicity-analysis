import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import DeepLift
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
test_text = "I kill niggers"

# Проверка predict_proba
probs = predict_proba([test_text])
print("Predicted probabilities:", probs)

# Генерация объяснения DeepLift
def generate_deeplift_explanation(text):
    try:
        # Токенизация текста
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Получаем эмбеддинги из модели
        with torch.no_grad():
            embeddings = model.get_input_embeddings()(input_ids)

        # Создаем класс-обертку для forward_func
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super(ModelWrapper, self).__init__()
                self.model = model

            def forward(self, embeddings, attention_mask=None):
                outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
                return outputs.logits

        # Обертываем модель
        wrapped_model = ModelWrapper(model)

        # Создаем объяснитель DeepLift
        deep_lift = DeepLift(wrapped_model)

        # Определяем базовые значения (reference values)
        baseline_embeddings = torch.zeros_like(embeddings)

        # Вычисляем атрибуции с помощью DeepLIFT
        attributions = deep_lift.attribute(
            inputs=embeddings,
            baselines=baseline_embeddings,
            additional_forward_args=(attention_mask,),
            target=0  # target=0 для токсичного класса
        )

        # Преобразуем атрибуции в читаемый формат
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)

        # Визуализируем результат
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
        print("DeepLift features:", dict(zip(merged_tokens, merged_attributions)))

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

        plt.title("DeepLift: Word Importance", fontsize=14, fontweight='bold')
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
        print(f"Error generating DeepLift explanation: {str(e)}")
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

# Генерируем объяснение DeepLift для тестового текста
deeplift_graph = generate_deeplift_explanation(test_text)
if deeplift_graph:
    print("DeepLift graph generated successfully.")
else:
    print("Failed to generate DeepLift graph.")