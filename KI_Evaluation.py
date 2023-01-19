import tensorflow as tf
from keras import layers
import numpy as np
import pandas as pd

# Daten einlesen
alzheimer_data = pd.read_csv("C:/Users/KDick/PycharmProjects/pythonProject1/inputs/Alzheimer/oasis_longitudinal.csv")

print('')
print('Das sind die eingelesenen Daten:\n', alzheimer_data.head())

# Die für den Aufbau nicht benötigten Spalten Subject ID MRI ID Group und Hand aus den Daten löschen.
# Hand wurde herausgenommen, da nur rechtshändige Personen in der Versuchsgruppe vertreten waren und die information
# somit irrelevant zur Auswertung der Daten wurde.
# Subject ID und MR ID wurden herausgenommen, da die rein kennzeichnenden Eigenschaften dieser Spalten nicht zur
# Klassifikation beitragen.
# Group wurde gelöscht, da diese Spalte das Ergebnis der Klassifikation vorwegnehmen würde.

alzheimer_data_clean = alzheimer_data.drop(['Subject ID', 'MRI ID', 'Group', 'Hand'], axis=1)
print('')
print('Das sind die eíngelesenen Daten ohne die Spalten Subject ID, MRI ID, Group und Hand:\n',
      alzheimer_data_clean.head())

alzheimer_data_clean['SES'].fillna((alzheimer_data_clean['SES'].mean()), inplace=True)
alzheimer_data_clean['MMSE'].fillna((alzheimer_data_clean['MMSE'].mean()), inplace=True)

print('')
print("Das sind die Daten ohne NaN:\n", alzheimer_data_clean.head())

alzheimer_data_clean.rename(columns={'MR Delay': 'mrdelay', 'Visit': 'visit', 'Age': 'age', 'EDUC': 'educ',
                                     'SES': 'ses', 'MMSE': 'mmse', 'CDR': 'cdr', 'eTIV': 'etiv', 'nWBV': 'nwbv',
                                     'ASF': 'asf', 'M/F': 'm_f'}, inplace=True)

print('')
print('Das sind die Daten mit umbenannten Spalten (für spätere Speicherung wichtig):\n', alzheimer_data_clean.head())

alzheimer_data_clean_shuffled = alzheimer_data_clean.sample(frac=1).reset_index(drop=True)

print('')
print('Das sind die Daten einmal durchgemischt:\n', alzheimer_data_clean_shuffled.head())

train, val, test = np.split(alzheimer_data_clean.sample(frac=1),
                            [int(0.8 * len(alzheimer_data_clean)), int(0.9 * len(alzheimer_data_clean))])

print('')
print(len(train), 'Trainingsdaten')
print('')
print(len(val), 'Validierungsdaten')
print('')
print(len(test), 'Testdaten')


def df_to_dataset(dataframe, shuffle=True, batch_size=64):
    df = dataframe.copy()
    labels = df.pop('cdr')
    df = {key: value[:, tf.newaxis] for key, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
[(train_features, label_batch)] = train_ds.take(1)
print('Jedes Feature:', list(train_features.keys()))
print('')
print('Ein Paar Altersangaben:', train_features['age'])
print('')
print('Ein Paar CDR Angaben:', label_batch)

# Definition der Methode zur Normierung der numerischen Werte.

def get_normalization_layer(name, dataset):
    normalizer = layers.Normalization(axis=None)

    feature_ds = dataset.map(lambda x, y: x[name])

    normalizer.adapt(feature_ds)

    return normalizer

# Definition der Methode zur Umwandlung der String Werte

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    if dtype == 'string':
        index = layers.StringLookup(max_tokens=max_tokens)

    feature_ds = dataset.map(lambda x, y: x[name])

    index.adapt(feature_ds)

    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

    return lambda feature: encoder(index(feature))


test_type_col = train_features['m_f']
test_type_layer = get_category_encoding_layer(name='m_f',
                                              dataset=train_ds,
                                              dtype='string')

train_ds = df_to_dataset(train, batch_size=298)
val_ds = df_to_dataset(val, shuffle=False, batch_size=37)
test_ds = df_to_dataset(test, shuffle=False, batch_size=38)

all_inputs = []
encoded_features = []

# Normierung der numerischen Werte.

numerical_features = ['visit', 'mrdelay', 'age', 'educ', 'ses', 'mmse', 'cdr', 'etiv', 'nwbv', 'asf']
for header in numerical_features:
    numeric_col = tf.keras.Input(shape=(1,), name=header)
    normalization_layer = get_normalization_layer(header, train_ds)
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs.append(numeric_col)
    encoded_features.append(encoded_numeric_col)

# Umwandlung der String Werte in numerische Werte.

sex_col = tf.keras.Input(shape=(1,), name='m_f', dtype='string')

encoding_layer = get_category_encoding_layer(name='m_f',
                                            dataset=train_ds,
                                            dtype='string',
                                            max_tokens=5)
encoded_sex_col = encoding_layer(sex_col)
all_inputs.append(sex_col)
encoded_features.append(encoded_sex_col)

all_features = tf.keras.layers.concatenate(encoded_features)

reloaded_model = tf.keras.models.load_model('Meine_Alzheimer_KI')

test_one = df_to_dataset(test, shuffle=True, batch_size=1)
[(test_features, test_label_batch)] = test_one.take(1)
print('\nDie Daten der Person die getestet werden:\n',test_features)
print('\nDie CDR Angabe der Person: ', test_label_batch)
predictions = reloaded_model.predict(test_one)
i=0.0
pred = 0.0
for value in predictions:
    if value >= 1.5:
        pred = pred + 2.0
        i = i + 1.0
    if 1.0 <= value < 1.5:
        pred = pred + 1.0
        i = i + 1.0
    if 0.5 <= value < 1.0:
        pred = pred + 0.5
        i = i + 1.0
    if value < 0.5:
        i = i + 1.0

print('\nDer Prediction Wert: ', pred, '\nDer i-Wert: ', i)

pred_est = pred/i

print('Diesen prediction Wert gibt die KI aus: ', pred_est)

if pred_est >= 1.5:
    print('Prediction für Testperson: CDR = 2.0')
if 1.0 <= pred_est < 1.5:
    print('Prediction für Testperson: CDR = 1.0')
if 0.5 <= pred_est < 1.0:
    print('Prediction für Testperson: CDR = 0.5')
if pred_est < 0.5:
    print('Prediction für Testperson: CDR = 0.0')
