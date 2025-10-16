import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import os
from transformers import AutoTokenizer
import numpy as np

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

class MultimodalDataset(Dataset):
    def __init__(self, dish_df, ingredients_df, image_dir, split='train', transform=None, max_length=128):
        self.dish_df = dish_df[dish_df['split'] == split].reset_index(drop=True)
        self.ingredients_df = ingredients_df
        self.image_dir = image_dir
        self.transform = transform
        self.max_length = max_length
        
        # токенайзер
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # создаем нормализованный маппинг ингредиентов
        self.ingr_to_idx = {}
        self.ingr_mapping = {}
        
        for idx, (_, row) in enumerate(ingredients_df.iterrows()):
            # нормализуем ID для сопоставления
            norm_id = str(row['id']).strip()
            self.ingr_to_idx[norm_id] = idx
            self.ingr_mapping[norm_id] = row['ingr']
        
        self.num_ingredients = len(self.ingr_to_idx)
        
        print(f"✅ Создан словарь с {self.num_ingredients} ингредиентами")
        
    def __len__(self):
        return len(self.dish_df)
    
    def _get_ingredient_text(self, ingredient_ids):
        """Конвертация ингредиента по ID в текст с нормализацией"""
        if pd.isna(ingredient_ids):
            return "no ingredients"
        
        ingr_list = ingredient_ids.split(';')
        ingr_names = []
        
        for ingr_id in ingr_list:
            # нормализуем ID перед поиском
            normalized_id = normalize_ingredient_id(ingr_id)
            ingr_name = self.ingr_mapping.get(normalized_id, "unknown")
            ingr_names.append(ingr_name)
        
        return ", ".join(ingr_names)
    
    def _get_multi_hot_ingredients(self, ingredient_ids):
        """Многомерное бинарное представление ингредиентов с нормализацией"""
        ingredients_vec = torch.zeros(self.num_ingredients)
        if pd.notna(ingredient_ids):
            for ingr_id in ingredient_ids.split(';'):
                # нормализуем ID перед поиском
                normalized_id = normalize_ingredient_id(ingr_id)
                if normalized_id in self.ingr_to_idx:
                    ingredients_vec[self.ingr_to_idx[normalized_id]] = 1
                else:
                    # логируем пропущенные ингредиенты для отладки
                    pass
        return ingredients_vec
    
    def __getitem__(self, idx):
        row = self.dish_df.iloc[idx]
        dish_id = str(row['dish_id'])
        
        # загрузка и преобразование фотографий
        img_path = os.path.join(self.image_dir, dish_id, 'rgb.png')
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # текстовая модальность: encoding ингредиентов
        ingredient_text = self._get_ingredient_text(row['ingredients'])
        text_encoding = self.tokenizer(
            ingredient_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # табличная модальность: multi-hot бинарная таблица с ингредиентами
        ingredients_tabular = self._get_multi_hot_ingredients(row['ingredients'])
        # проверка, что хотя бы некоторые ингредиенты найдены
        if ingredients_tabular.sum() == 0 and pd.notna(row['ingredients']):
            print(f"⚠️ Предупреждение: для блюда {dish_id} не найдены ингредиенты")
        
        # признак массы
        mass = torch.tensor(row['total_mass'], dtype=torch.float32)
        
        # целевое
        calories = torch.tensor(row['total_calories'], dtype=torch.float32)
        
        return {
            'image': image,
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'ingredients_tabular': ingredients_tabular,
            'mass': mass,
            'calories': calories,
            'dish_id': dish_id
        }

def collate_fn(batch):
    """кастомная collate function для мультимодальной модели"""
    batch_dict = {}
    for key in batch[0].keys():
        if key in ['input_ids', 'attention_mask']:
            batch_dict[key] = torch.stack([item[key] for item in batch])
        elif key in ['image', 'ingredients_tabular', 'mass', 'calories']:
            batch_dict[key] = torch.stack([item[key] for item in batch])
        else:
            batch_dict[key] = [item[key] for item in batch]
    return batch_dict

def get_transforms(mode='train'):
    """преобразования фотографий для train/validation"""
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def prepare_data(cfg):
    """подготовка и загрузка исходного датасета"""
    # загрузка данных
    dish_df = pd.read_csv(f"{cfg.DATA_PATH}/dish.csv")
    ingredients_df = pd.read_csv(f"{cfg.DATA_PATH}/ingredients.csv")
    
    # тщательная проверка сопоставления
    #print("🔍 Детальная проверка сопоставления ингредиентов...")
    
    # Преобразуем ID в ingredients_df к строковому типу без пробелов
    ingredients_df['id_str'] = ingredients_df['id'].astype(str).str.strip()
    
    test_ingredients = dish_df['ingredients'].dropna().iloc[0]
    ingr_list = test_ingredients.split(';')
    
    found_count = 0
    not_found_count = 0
    
    for ingr_id in ingr_list:
        normalized_id = normalize_ingredient_id(ingr_id)
        
        # Пробуем несколько стратегий поиска
        found = False
        
        # Стратегия 1: Прямое совпадение с normalized_id
        ingr_row = ingredients_df[ingredients_df['id_str'] == normalized_id]
        if not ingr_row.empty:
            #print(f"✅ {ingr_id} -> {normalized_id} -> {ingr_row.iloc[0]['ingr']}")
            found = True
        else:
            # Стратегия 2: Ищем среди всех ID в ingredients_df
            for idx, row in ingredients_df.iterrows():
                if str(row['id_str']).strip() == normalized_id:
                    #print(f"✅ {ingr_id} -> {normalized_id} -> {row['ingr']}")
                    found = True
                    break
        
        if found:
            found_count += 1
        else:
            print(f"❌ {ingr_id} -> {normalized_id} -> НЕ НАЙДЕН")
            # Дополнительная отладка
            print(f"   Доступные ID вокруг {normalized_id}:")
            similar_ids = ingredients_df[
                ingredients_df['id_str'].str.startswith(normalized_id[:2]) |
                ingredients_df['id_str'].str.endswith(normalized_id[-2:])
            ].head(3)
            if not similar_ids.empty:
                for _, row in similar_ids.iterrows():
                    print(f"   📍 {row['id_str']}: {row['ingr']}")
            not_found_count += 1
    
    #print(f"\n📊 Статистика сопоставления: {found_count} найдено, {not_found_count} не найдено")
    
    # удаление ошибочных данных
    dish_df['calories_per_gram'] = dish_df['total_calories'] / dish_df['total_mass']
    clean_dish_df = dish_df[(dish_df['calories_per_gram'] >= 0.2) & (dish_df['calories_per_gram'] <= 8)]
    
    print(f"Оригинальный датасет: {len(dish_df)} примеров")
    print(f"После очистки: {len(clean_dish_df)} примеров")
    
    # создание трайн и тестового датасетов
    train_dataset = MultimodalDataset(
        clean_dish_df, ingredients_df, cfg.IMAGE_DIR, 
        split='train', transform=get_transforms('train')
    )
    
    test_dataset = MultimodalDataset(
        clean_dish_df, ingredients_df, cfg.IMAGE_DIR, 
        split='test', transform=get_transforms('val')
    )
    
    return train_dataset, test_dataset, len(ingredients_df)