import pickle
import math
import re
import pandas as pd

# тут функция, которая сегментирует текст без пробелов
# на основе вероятностей слов из словаря word_probs
# работает на логарифмах вероятностей, чтобы значения не взрывались
# и на дп

def primary_segment(text, word_probs, max_word_length=20):
    n = len(text)
    best = [(0.0, [])] + [(float('-inf'), [])] * n
    for i in range(1, n + 1):
        for j in range(max(0, i - max_word_length), i):
            word = text[j:i]
            if word in word_probs:
                prob = best[j][0] + math.log(word_probs[word])
                if prob > best[i][0]:
                    best[i] = (prob, best[j][1] + [word])
    
    return best[n][1]

# тут функция, которая считает индексы пробелов в тексте по условиям из степика

def calculate_indexes(text):
    result = []
    counter = 0
    for i, item in enumerate(text):
        if item == " ":
            result.append(counter)
        else:
            counter += 1
    return result

# тут мега крутая функция, принимает на вход текст без пробелов, 
# но с спец, и текст с пробелами, возвращает текст с пробелами и спец символами

def return_spec(text, text_with_spec, spec_dict):
    result = []
    text_i = 0
    spec_i = 0

    def is_word_char(ch):
        return re.match(r'[а-яёa-z0-9]', ch, re.IGNORECASE) is not None

    while text_i < len(text) and spec_i < len(text_with_spec):
        ch_text = text[text_i]
        ch_spec = text_with_spec[spec_i]

        if ch_text == " ":
            result.append(" ")
            text_i += 1
            continue

        if not is_word_char(ch_spec):
            result.append(spec_dict.get(ch_spec, ch_spec + " "))
            spec_i += 1
            continue

        if ch_text.lower() == ch_spec.lower():
            result.append(ch_text)
            text_i += 1
            spec_i += 1
        else:
            spec_i += 1

    while text_i < len(text):
        result.append(text[text_i])
        text_i += 1

    while spec_i < len(text_with_spec):
        ch_spec = text_with_spec[spec_i]
        if not is_word_char(ch_spec):
            result.append(spec_dict.get(ch_spec, ch_spec + " "))
        spec_i += 1

    return re.sub(' +', ' ', ''.join(result)).strip()

# захендлил спец символы

def handle_special_characters(segmented, text):
    final_text = " ".join(segmented)
    final_text = return_spec(final_text, text, {"-": " - ", "!": "! ", "—" : " — "})
    final_text = re.sub(' , ', ', ', final_text).strip()
    final_text = re.sub(' \. ', '. ', final_text).strip()
    final_text = re.sub(' _ ', '_', final_text).strip()

# просто все воедино собрал

def calculate_result(text):
    text_without_spec = re.sub(r'[^а-яёa-z0-9]+', '', text.lower())
    segmented = primary_segment(text_without_spec, word_probs)
    final_text = handle_special_characters(segmented, text)
    return calculate_indexes(final_text), final_text

# тут я загружаю словарь с вероятностями слов
# я посчитал его в word_prob_creation.ipynb

probs_path = "probs.pkl"
with open(probs_path, 'rb') as f:
    word_probs = pickle.load(f)

# тут снизу я загружаю тестовый датасетик 
# и создаю submission.csv, можно поменять название файла

data = []
with open('dataset_1937770_3.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split(',', 1)
            if len(parts) == 2:
                data.append({
                    'id': parts[0],
                    'text': parts[1]
                })
    
df = pd.DataFrame(data)
df['predicted_positions'] = df['text'].apply(lambda t: str(calculate_result(t)[1]))
submission = df[['id', 'predicted_positions']].copy()
submission = submission.iloc[1:].reset_index(drop=True)
submission.to_csv('submission.csv', index=False, encoding='utf-8')