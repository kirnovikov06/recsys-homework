# Данные

Новиков Кирилл Сергеевич, группа Б05-307.

Amazon Product Data (2014), Video Games, 5-core: 231 780 отзывов.

Первоисточник: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html

Отзывы: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz

Метаданные: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Video_Games.json.gz

Данные загружает `load_data()` из `utils.py`. Хеши исходных архивов, фактические размеры и этапы очистки выводятся в ноутбуке.

Берутся 3 000 товаров с наибольшим числом отзывов до первой временной границы. При равенстве счётчиков товары сортируются по ASIN. После этого остаются пользователи хотя бы с тремя обучающими взаимодействиями. Будущие оценки и частоты в отборе не используются. Ограничение каталога позволяет обучить EASE на обычном компьютере. Полный протокол, ограничения и использованные поля описаны в ноутбуке.
