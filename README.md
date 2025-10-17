# calories_dish
AI solution. Get answer the question: How many calories are in the dish?

Датасет: nutrition.zip
Владелец:upskilling.production
Размер:1,32 ГБ
Скачен по ссылке: https://disk.yandex.ru/d/kz9g5msVqtahiw

Отдельные функции в папке script в файлах dataset.py, utils.py, evaluation.py
Пайплайн преданализа датасета, обучения и тестирования модели в файле calories_solution.ipynb

Обучение на CPU (5 эпох): calories_solution.ipynb
ФИНАЛЬНЫЕ МЕТРИКИ МОДЕЛИ
============================================================
📊 Test MAE:  48.30 ккал
📊 Test RMSE: 73.94 ккал
📊 Test R²:   0.8780
📊 Test F1:   0.7981
🎯 ✅ Модель соответствует целевому показателю (MAE < 50)

Обучение модели на GPU (30 эпох): calories_solution_from_gpu.ipynb. 
ФИНАЛЬНЫЕ МЕТРИКИ МОДЕЛИ
============================================================
📊 Test MAE:  46.11 ккал
📊 Test RMSE: 73.01 ккал
📊 Test R²:   0.8811
📊 Test F1:   0.8095
🎯 ✅ Модель соответствует целевому показателю (MAE < 50)
