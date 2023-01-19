import tensorflow as tf
from keras import layers, regularizers
import numpy as np
import pandas as pd

# numpy Inhalte leserlicher gestalten
np.set_printoptions(precision=3, suppress=True)

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

i = 0
accuracy_add = 0.0
accuracy_best = 0.0
accuracy_worst = 1.0

while i <= 100:
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

    print('Durchgang Nummer: ', i)

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

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001, decay_steps= 15, decay_rate=1, staircase=False )
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    x = tf.keras.layers.Dense(20, activation="tanh",
                              kernel_regularizer=regularizers.l2(0.001))(all_features)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, kernel_regularizer=regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, kernel_regularizer=regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dense(64, kernel_regularizer=regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dense(32, kernel_regularizer=regularizers.l2(0.001))(x)
    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(all_inputs, output)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.Huber(),
                  metrics=["accuracy"])

    model.fit(train_ds, epochs=298, validation_data=val_ds)
    loss, accuracy = model.evaluate(test_ds)

    if accuracy > accuracy_best:
        accuracy_best = accuracy
        model.save('Meine_Alzheimer_KI')

    if accuracy < accuracy_worst:
        accuracy_worst = accuracy

    accuracy_add = accuracy_add + accuracy
    i = i + 1

print("Der i-Wert ", i)
print("Addierte Accuracy: ", accuracy_add)
accuracy_average = accuracy_add / (i-1)

print("Durchschnittliche Accuracy:", accuracy_average)
print("Beste Accuracy: ", accuracy_best)
print("Schlechteste Accuracy: ", accuracy_worst)


