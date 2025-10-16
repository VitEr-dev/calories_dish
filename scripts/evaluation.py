import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

def display_top_errors(model, test_dataset, predictions, targets, cfg, top_k=5):
    """отображение топ блюд с наибольшей ошибкой предсказания"""
    
    # загружаем исходные данные для получения информации о блюдах
    dish_df = pd.read_csv(f"{cfg.DATA_PATH}/dish.csv")
    ingredients_df = pd.read_csv(f"{cfg.DATA_PATH}/ingredients.csv")
    
    #print(f"🔍 Загружено {len(ingredients_df)} ингредиентов")
    #print("Примеры из ingredients.csv:")
    #print(ingredients_df.head(3))
    
    # вычисляем ошибки для каждого примера
    errors = np.abs(np.array(predictions) - np.array(targets))
    
    # получаем индексы топ-K наибольших ошибок
    top_error_indices = np.argsort(errors)[-top_k:][::-1]
    
    print(f"\n{'='*60}")
    print(f"🍽️  ТОП-{top_k} БЛЮД С НАИБОЛЬШЕЙ ОШИБКОЙ ПРЕДСКАЗАНИЯ")
    print(f"{'='*60}")
    
    for i, idx in enumerate(top_error_indices, 1):
        # получаем данные примера
        sample = test_dataset[idx]
        dish_id = sample['dish_id']
        
        # находим информацию о блюде в DataFrame
        dish_info = dish_df[dish_df['dish_id'].astype(str) == dish_id].iloc[0]
        
        actual_calories = targets[idx]
        predicted_calories = predictions[idx]
        error = errors[idx]
        
        # получаем названия ингредиентов
        ingredient_text = get_ingredient_names(dish_info['ingredients'], ingredients_df)
        
        print(f"\n🔴 #{i} | Ошибка: {error:.1f} ккал")
        print(f"   ID блюда: {dish_id}")
        print(f"   🎯 Фактические калории: {actual_calories:.0f} ккал")
        print(f"   🤖 Предсказанные калории: {predicted_calories:.0f} ккал")
        print(f"   ⚖️  Масса блюда: {dish_info['total_mass']:.0f} г")
        print(f"   🥗 Ингредиенты: {ingredient_text}")
        
        # отображаем изображение
        img_path = os.path.join(cfg.IMAGE_DIR, dish_id, 'rgb.png')
        if os.path.exists(img_path):
            try:
                plt.figure(figsize=(8, 6))
                img = Image.open(img_path)
                plt.imshow(img)
                plt.title(f'Блюдо #{i}\nФакт: {actual_calories:.0f} ккал | Предсказание: {predicted_calories:.0f} ккал\nОшибка: {error:.1f} ккал', 
                         fontsize=12, pad=20)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"   ❌ Ошибка загрузки изображения: {e}")
        else:
            print(f"   ❌ Изображение не найдено: {img_path}")

def analyze_error_distribution(predictions, targets):
    """Анализ распределения ошибок"""
    errors = np.array(predictions) - np.array(targets)
    absolute_errors = np.abs(errors)
    
    plt.figure(figsize=(15, 5))
    
    # Распределение ошибок
    plt.subplot(1, 3, 1)
    plt.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Ошибка предсказания (ккал)')
    plt.ylabel('Количество примеров')
    plt.title('Распределение ошибок\n(Отрицательные = недооценка)')
    
    # Абсолютные ошибки
    plt.subplot(1, 3, 2)
    plt.hist(absolute_errors, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Целевой MAE')
    plt.xlabel('Абсолютная ошибка (ккал)')
    plt.ylabel('Количество примеров')
    plt.title('Распределение абсолютных ошибок')
    plt.legend()
    
    # Предсказания vs Фактические значения
    plt.subplot(1, 3, 3)
    plt.scatter(targets, predictions, alpha=0.6, s=20)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', linewidth=2)
    plt.xlabel('Фактические калории')
    plt.ylabel('Предсказанные калории')
    plt.title('Предсказания vs Фактические значения')
    
    plt.tight_layout()
    plt.show()
    
    # Статистика ошибок
    print(f"\n{'='*50}")
    print("СТАТИСТИКА ОШИБОК")
    print(f"{'='*50}")
    print(f"📈 Средняя ошибка: {np.mean(errors):.2f} ккал")
    print(f"📈 Медианная ошибка: {np.median(errors):.2f} ккал")
    print(f"📈 Стандартное отклонение: {np.std(errors):.2f} ккал")
    print(f"📊 % примеров с ошибкой < 50 ккал: {np.mean(absolute_errors < 50) * 100:.1f}%")
    print(f"📊 % примеров с ошибкой < 100 ккал: {np.mean(absolute_errors < 100) * 100:.1f}%")
    
def normalize_ingredient_id(ingr_id):
    """Нормализация ID ингредиента - несколько стратегий поиска"""
    if pd.isna(ingr_id):
        return ingr_id
    
    ingr_id = str(ingr_id).strip()
    
    # убираем 'ingr_' префикс
    if ingr_id.startswith('ingr_'):
        simple_normalized = ingr_id[5:]  # Убираем 'ingr_'
        
        # убираем ведущие нули после префикса
        if simple_normalized.isdigit():
            digit_normalized = str(int(simple_normalized))
            return digit_normalized
        return simple_normalized
    
    return ingr_id

def get_ingredient_names(ingredient_ids, ingredients_df):
    """Получение названий ингредиентов по их ID с нормализацией"""
    if pd.isna(ingredient_ids) or ingredient_ids == '':
        return "Нет ингредиентов"
    
    ingr_list = ingredient_ids.split(';')
    ingr_names = []
    
    # создаем нормализованный словарь для быстрого поиска
    ingr_dict = {}
    for _, row in ingredients_df.iterrows():
        # преобразуем ID в строку и убираем лишние пробелы
        normalized_id = str(row['id']).strip()
        ingr_dict[normalized_id] = row['ingr']
    
    #print(f"🔍 Создан словарь с {len(ingr_dict)} ингредиентами")
    #print("Примеры из словаря:")
    #for i, (ingr_id, ingr_name) in enumerate(list(ingr_dict.items())[:5]):
        #print(f"  '{ingr_id}' -> '{ingr_name}'")
    
    for ingr_id in ingr_list:
        # нормализуем ID из dish.csv
        normalized_id = normalize_ingredient_id(ingr_id.strip())
        
        if normalized_id in ingr_dict:
            ingr_names.append(ingr_dict[normalized_id])
        else:
            # пробуем найти без нормализации
            if ingr_id in ingr_dict:
                ingr_names.append(ingr_dict[ingr_id])
            else:
                ingr_names.append(f"неизвестный ({ingr_id} -> {normalized_id})")
    
    return ", ".join(ingr_names)

