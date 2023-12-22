import pandas as pd
from preprocessing import first_preprocess

# Загрузка данных
df = pd.read_csv('./data/train_df.csv')

# Первичная обработка
df = first_preprocess(df)


from preprocessing import preprocess_data

if __name__ == '__main__':
    df = preprocess_data(df)


from catboost import CatBoostClassifier
from catboost import Pool


def fit_model(train_pool, validation_pool, **kwargs):
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.009,
        eval_metric='MultiClass',
        #early_stopping_rounds=30,
        use_best_model= True,
        task_type='CPU',
        **kwargs
    )

    return model.fit(
        train_pool,
        eval_set=validation_pool,
        verbose=100,
    )

from sklearn.model_selection import train_test_split as tts
df.reset_index(drop=True, inplace=True)

df_train_val = df[['bank', 'feeds', 'lemmas', 'year', 'month', 'day', 'time_day', 'sym_len', 'word_len']]
y_train_val = df['grades']
X_train, X_val, y_train, y_val = tts(df_train_val, y_train_val, shuffle=True, stratify=y_train_val, train_size=0.999)


train_pool = Pool(
    X_train, y_train, 
    cat_features=['bank', 'time_day', 'year', 'month', 'day'],
    text_features=['lemmas', 'feeds'],
)

validation_pool = Pool(
    X_val, y_val, 
    cat_features=['bank', 'time_day', 'year', 'month', 'day'],
    text_features=['lemmas', 'feeds'],
)

print('Train dataset shape: {}\n'.format(train_pool.shape))

model = fit_model(train_pool, validation_pool)
model.save_model('model.bin')