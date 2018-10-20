import numpy as np
import pandas as pd
# объявим где хранятся исходные данные
PATH_TRAIN = '../input/train.csv'
PATH_TEST = '../input/test.csv'

# объявим куда сохраним результат
PATH_PRED = 'pred.csv'

## Из тренировочного набора собираем статистику о встречаемости слов
# создаем словарь для хранения статистики
word_stat_dict = {}

# открываем файл на чтение в режиме текста
fl = open(PATH_TRAIN, 'rt', encoding='utf-8')

# считываем первую строчку - заголовок (она нам не нужна)
fl.readline()

# в цикле читаем строчки из файла

X = []
y = []
for line in fl:
    # разбиваем строчку на три строковые переменные
    Id, Sample, Prediction = line.strip().split(',')
    X.append (Sample)
    y.append (Prediction)

# закрываем файл
fl.close ()

X_train = np.array (X)
y_train = np.array (y)

# заполняем словари sample_words_dict и most_often_word_for_sample
all_words_dict = {}
sample_words_dict = {}
for i in range (y_train.size):
    ans = y_train[i]
    if ans not in all_words_dict:
        all_words_dict[ans] = 0
    all_words_dict[ans] += 1

    sample = X_train[i]
    if sample not in sample_words_dict:
        sample_words_dict[sample] = {}

    if ans not in sample_words_dict[sample]:
        sample_words_dict[sample][ans] = 0
    sample_words_dict[sample][ans] += 1

most_often_word_for_sample = {}
for sample in sample_words_dict:
    most_often_word_for_sample[sample] = max (sample_words_dict[sample], key=sample_words_dict[sample].get)

# открываем файл на чтение в режиме текста
fl = open(PATH_TEST, 'rt', encoding='utf-8')

# считываем первую строчку - заголовок (она нам не нужна)
fl.readline()

# открываем файл на запись в режиме текста
out_fl = open(PATH_PRED, 'wt', encoding='utf-8')

# записываем заголовок таблицы
out_fl.write('Id,Prediction\n')

def prediction_by_sample (x):
    x_pred = x
    if x in most_often_word_for_sample:
        x_pred = most_often_word_for_sample[x]
    else:
        max = 0
        for word in all_words_dict:
            if word.find(x) == 0 and all_words_dict[word] > max:
                x_pred = word
    return x_pred


# в цикле читаем строчки из тестового файла
for line in fl:
    # разбиваем строчку на две строковые переменные
    Id, Sample = line.strip().split(',')
    x_pred = prediction_by_sample(Sample)
    out_fl.write ('%s,%s\n' % (Id, x_pred))

fl.close()
out_fl.close()