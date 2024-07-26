# Pose encoder

## Описание

Базовый класс, кодирующий скелет в своё пространство, 
и его наследник, кодирующий конкретно в UMAP.

### Пример использования для кодирования поз

```Python
import numpy as np
import random
from umap_landmarks_encoder import UMAPLandmarksEncoder

# Определяем индексы для части тела, которую хотим закодировать (например, верхняя часть тела)
upper_body_indices = [4, 5, 6]

# Инициализируем энкодер
poseEncoder = UMAPLandmarksEncoder(input_landmarks_shape=(len(upper_body_indices), 2), embedding_shape=(2,))

# Упрощенная загрузка набора данных (похоже на fit_skeletons_umap.py)
def load_sample_data(file_path):
    # Загружаем данные полного скелета
    full_skeletons = np.load(file_path)
    # Извлекаем только ключевые точки верхней части тела
    return full_skeletons[:, upper_body_indices, :]

# Загружаем набор данных
dataset = load_sample_data("path/to/your/skeleton_data.npy")

# Добавляем позы в энкодер
for pose in dataset:
    poseEncoder.add(pose)

# Обучаем энкодер
poseEncoder.fit()

# Кодируем случайный образец из набора данных
random_pose = random.choice(dataset)
embedding = poseEncoder.encode(random_pose)

print(f"Закодированное представление: {embedding}")

# Примечание: Декодирование не реализовано в текущей версии
# Если бы оно было реализовано, вы могли бы использовать его так:
# decoded_pose = poseEncoder.decode(embedding)
# print(f"Декодированная поза: {decoded_pose}")

# Сохраняем энкодер
poseEncoder.save("upper_body_encoder.pkl")

# Загружаем энкодер (в новой сессии или скрипте)
loaded_encoder = UMAPLandmarksEncoder(input_landmarks_shape=(len(upper_body_indices), 2), embedding_shape=(2,))
loaded_encoder.load("upper_body_encoder.pkl")

# Используем загруженный энкодер
new_pose = dataset[0]  # Просто используем первую позу в качестве примера
new_embedding = loaded_encoder.encode(new_pose)
print(f"Новое закодированное представление: {new_embedding}")
```

### Пример использования для генерации датасета

`python generate_bedlam_based_dataset.py`

## Установка

`pip install -r requirements.txt`