import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
import timm
import random
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
import warnings
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import MultimodalDataset, collate_fn, get_transforms, prepare_data
warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    """для воспроизведения экспериментов"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_requires_grad(model, requires_grad):
    """Set requires_grad for model parameters"""
    for param in model.parameters():
        param.requires_grad = requires_grad

def unfreeze_layers(model, layer_names):
    """разморозка слоев"""
    if not layer_names:
        return
    
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True

class MultimodalModel(nn.Module):
    def __init__(self, num_ingredients, cfg):
        super().__init__()
        self.cfg = cfg
        
        # текстовая модальность: модель BERT
        self.text_model = AutoModel.from_pretrained(cfg.TEXT_MODEL_NAME)
        set_requires_grad(self.text_model, False)  # заморозка на старте
        if cfg.TEXT_MODEL_UNFREEZE:
            unfreeze_layers(self.text_model, cfg.TEXT_MODEL_UNFREEZE)
        self.text_proj = nn.Linear(768, cfg.HIDDEN_DIM)
        
        # фото модальность: модель EfficientNet
        self.image_model = timm.create_model(cfg.IMAGE_MODEL_NAME, pretrained=True, num_classes=0)
        set_requires_grad(self.image_model, False)  # заморозка на старте
        if cfg.IMAGE_MODEL_UNFREEZE:
            unfreeze_layers(self.image_model, cfg.IMAGE_MODEL_UNFREEZE)
        self.image_proj = nn.Linear(self.image_model.num_features, cfg.HIDDEN_DIM)
        
        # табличная модальность: multi-hot инградиентов
        self.ingredient_encoder = nn.Sequential(
            nn.Linear(num_ingredients, 512),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT)
        )
        
        # признак массы
        self.mass_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT)
        )
        
        # слияние слоев
        self.fusion = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM * 2 + 128 + 64, 512),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(128, 1)  # предсказание - результат регрессии
        )
        
    def forward(self, input_ids, attention_mask, image, ingredients_tabular, mass):
        # текстовые признаки
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] токен
        text_features = self.text_proj(text_features)
        
        # признаки изображения
        image_features = self.image_model(image)
        image_features = self.image_proj(image_features)
        
        # табличные признаки
        ingredient_features = self.ingredient_encoder(ingredients_tabular)
        
        # признак массы
        mass_features = self.mass_encoder(mass.unsqueeze(1))
        
        # конкетинация всех признаков
        combined_features = torch.cat([
            text_features, 
            image_features, 
            ingredient_features, 
            mass_features
        ], dim=1)
        
        # финальное предсказание
        calories_pred = self.fusion(combined_features).squeeze()
        
        return calories_pred
    
def calculate_f1_score(preds, targets, threshold=50):
    """метрика F1 score для задачи регрессии с использованием пороговой калссификации"""
    # конвертируем в numpy
    preds = np.array(preds)
    targets = np.array(targets)
    
    # считаем absolute errors
    errors = np.abs(preds - targets)
    
    # бинарная классификация: 1 if error <= threshold, 0 otherwise
    pred_binary = (errors <= threshold).astype(int)
    true_binary = np.ones_like(pred_binary)  # все должно быть 1 в идельном случае
    
    # считаем метрику F1 score
    f1 = f1_score(true_binary, pred_binary, zero_division=0)
    return f1

def train_epoch(model, train_loader, optimizer, criterion, device, cfg):
    """цикл обучения для одной epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch in train_loader:
        # перенос данных на устройство device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image = batch['image'].to(device)
        ingredients_tabular = batch['ingredients_tabular'].to(device)
        mass = batch['mass'].to(device)
        calories = batch['calories'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        pred_calories = model(input_ids, attention_mask, image, ingredients_tabular, mass)
        
        # Calculate loss
        loss = criterion(pred_calories, calories)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if cfg.GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        
        # Optimizer step
        optimizer.step()
        
        # статистика
        running_loss += loss.item()
        all_preds.extend(pred_calories.detach().cpu().numpy())
        all_targets.extend(calories.cpu().numpy())
    
    # расчет метрик
    train_loss = running_loss / len(train_loader)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    f1 = calculate_f1_score(all_preds, all_targets)
    
    return train_loss, mae, rmse, r2, f1

def validate_epoch(model, val_loader, criterion, device, cfg):
    """цикл валидации для одной epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            # перенос на устройство device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image = batch['image'].to(device)
            ingredients_tabular = batch['ingredients_tabular'].to(device)
            mass = batch['mass'].to(device)
            calories = batch['calories'].to(device)
            
            # Forward pass
            pred_calories = model(input_ids, attention_mask, image, ingredients_tabular, mass)
            
            # Calculate loss
            loss = criterion(pred_calories, calories)
            
            # статистика
            running_loss += loss.item()
            all_preds.extend(pred_calories.cpu().numpy())
            all_targets.extend(calories.cpu().numpy())
    
    # расчет метрик
    val_loss = running_loss / len(val_loader)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    f1 = calculate_f1_score(all_preds, all_targets)
    
    return val_loss, mae, rmse, r2, f1, all_preds, all_targets

def create_optimizer(model, cfg):
    """создаем optimizer с разными значениями learning rates для разных слоев и моделей"""
    # разделяем параметры
    text_params = [p for n, p in model.named_parameters() if 'text' in n and p.requires_grad]
    image_params = [p for n, p in model.named_parameters() if 'image' in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if 'text' not in n and 'image' not in n and p.requires_grad]
    
    # создаем группы параметров для различных значений learning rates
    param_groups = [
        {'params': text_params, 'lr': cfg.TEXT_LR},
        {'params': image_params, 'lr': cfg.IMAGE_LR},
        {'params': other_params, 'lr': cfg.CLASSIFIER_LR}
    ]
    
    return torch.optim.AdamW(param_groups)

def train(cfg, device):
    """Основная функция обучения"""
    seed_everything(cfg.SEED)
    
    # загружаем и предобрабатываем данные из сырого датасета
    train_dataset, test_dataset, num_ingredients = prepare_data(cfg)
    
    # создаем data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    print(f"Примеров в Train: {len(train_dataset)}")
    print(f"Примеров в Test: {len(test_dataset)}")
    
    # инициализируем модель
    model = MultimodalModel(num_ingredients, cfg).to(device)
    
    # Loss/optimizer
    criterion = nn.L1Loss()  # MAE потери по регрессии
    optimizer = create_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # история обучения
    history = {
        'train_loss': [], 'train_mae': [], 'train_rmse': [], 'train_r2': [], 'train_f1': [],
        'test_loss': [], 'test_mae': [], 'test_rmse': [], 'test_r2': [], 'test_f1': []
    }
    
    print("Стартую обучение ...")
    best_mae = float('inf')
    patience_counter = 0
    
    for epoch in range(cfg.EPOCHS):
        # тренировка
        train_loss, train_mae, train_rmse, train_r2, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device, cfg
        )
        
        # валидация
        test_loss, test_mae, test_rmse, test_r2, test_f1, test_preds, test_targets = validate_epoch(
            model, test_loader, criterion, device, cfg
        )
        
        # Update scheduler
        scheduler.step(test_mae)
        
        # сохранение истории
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['train_rmse'].append(train_rmse)
        history['train_r2'].append(train_r2)
        history['train_f1'].append(train_f1)
        history['test_loss'].append(test_loss)
        history['test_mae'].append(test_mae)
        history['test_rmse'].append(test_rmse)
        history['test_r2'].append(test_r2)
        history['test_f1'].append(test_f1)
        
        # прогресс по эпохам в разрезе метрик
        print(f"Epoch {epoch+1}/{cfg.EPOCHS}:")
        print(f"  Train - MAE: {train_mae:.2f}, F1: {train_f1:.4f}, R²: {train_r2:.4f}")
        print(f"  Test  - MAE: {test_mae:.2f}, F1: {test_f1:.4f}, R²: {test_r2:.4f}")
        
        # ранняя остановка
        if test_mae < best_mae:
            best_mae = test_mae
            patience_counter = 0
            torch.save(model.state_dict(), cfg.SAVE_PATH)
        else:
            patience_counter += 1
            
        if patience_counter >= cfg.PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # загрузка лучшей модели и финальная оценка
    model.load_state_dict(torch.load(cfg.SAVE_PATH))
    final_loss, final_mae, final_rmse, final_r2, final_f1, final_preds, final_targets = validate_epoch(
        model, test_loader, criterion, device, cfg
    )
    
    print(f"\n{'='*50}")
    print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
    print(f"{'='*50}")
    print(f"Test MAE: {final_mae:.2f}")
    print(f"Test RMSE: {final_rmse:.2f}") 
    print(f"Test R²: {final_r2:.4f}")
    print(f"Test F1: {final_f1:.4f}")
    
    # проверка на соответствие результата исходной задаче mae < 50
    if final_mae < 50:
        print("✅ Приемлемый результат (MAE < 50)")
    else:
        print("❌ НЕ приемлемый результат")
    
    # визуализация результатов
    plot_results(history, final_preds, final_targets, final_mae)
    
    return model, history, final_mae, final_f1

def plot_results(history, preds, targets, final_mae):
    """Визуализация результатов обучения и тестирования"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: История обучения
    plt.subplot(1, 3, 1)
    plt.plot(history['train_mae'], label='Train MAE')
    plt.plot(history['test_mae'], label='Test MAE')
    plt.axhline(y=50, color='r', linestyle='--', label='Target MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('История обучения')
    
    # Plot 2: Предсказания vs Факт
    plt.subplot(1, 3, 2)
    plt.scatter(targets, preds, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('Фактические Calories')
    plt.ylabel('Предсказанные Calories')
    plt.title(f'Предсказанные vs Фактические\nMAE: {final_mae:.2f}')
    
    # Plot 3: F1 Score метрика
    plt.subplot(1, 3, 3)
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['test_f1'], label='Test F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('F1 Score метрика')
    
    plt.tight_layout()
    plt.show()