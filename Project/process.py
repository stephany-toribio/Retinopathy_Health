# instalar y configurar iniical
!pip install --upgrade tensorflow_hub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Descargar y descomprimir los datos
!pip install gdown h5py
!gdown --id '18HjAaZb26XCufJm5H-fD99-Z0j2u_0LS' -O upch-ml.zip
!unzip upch-ml.zip

# Directorios de datos
dir_data = '/content/train.csv'
dir_images = '/content/train/train/'
dir_images2 = '/content/test/test/'
dir_data2 = '/content/sample.csv'

# Leer y procesar los datos

def cargar_datos(dir_data, dir_images):
    data_df = pd.read_csv(dir_data)
    if 'location' in data_df.columns:
        data_df['ID'] = data_df['ID'] + '_' + data_df['location']
    data_df['path'] = dir_images + data_df['ID'] + '.jpg'
    data_df['exists'] = data_df['path'].map(os.path.exists)
    data_df.rename(columns={'level': 'score'}, inplace=True)
    data_df = data_df[data_df['exists']]
    data_df.drop(columns=['exists'], inplace=True)
    data_df.dropna(inplace=True)
    return data_df
data_df1 = cargar_datos(dir_data, dir_images)
data_df2 = cargar_datos(dir_data2, dir_images2)

#unfiifcar etiquetas y combinar datos
# No hay necesidad de unificar etiquetas si solo tienes 0 y 1
def unificar_etiquetas(data_df):
    return data_df

data_df1 = unificar_etiquetas(data_df1)
data_df2 = unificar_etiquetas(data_df2)
data_df2 = data_df2[data_df2['score'] == 1]

# Concatenar los datos
data_df = pd.concat([data_df1, data_df2], ignore_index=True)
#diviidir datos
# Dividir los datos en conjuntos de entrenamiento, validación y prueba
train_data, temp_data = train_test_split(data_df, test_size=0.3, random_state=42, stratify=data_df['score'])
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data['score'])

# Preprocesamiento de imágenes
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if mask.any():
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def load_ben_color(image):
    sigmaX = 50
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (200, 200))#224,224
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), 0, 100)#-4 128
    return image
#confiugrar generador de imagenes y parametros de entrenamiento
# Limpiar sesión de Keras
tf.keras.backend.clear_session()

# Convertir las etiquetas a tipo string
train_data['score'] = train_data['score'].astype(str)
val_data['score'] = val_data['score'].astype(str)
test_data['score'] = test_data['score'].astype(str)

# Parámetros de entrenamiento
epocas = 12
longitud, altura = 200,200
batch_size = 32

# Generadores de datos con aumento
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    preprocessing_function=load_ben_color
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    preprocessing_function=load_ben_color
)

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    preprocessing_function=load_ben_color
)
#creador generadores de datos
train_generator = train_datagen.flow_from_dataframe(
    train_data,
    x_col='path',
    y_col='score',
    target_size=(longitud, altura),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_dataframe(
    val_data,
    x_col='path',
    y_col='score',
    target_size=(longitud, altura),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_dataframe(
    test_data,
    x_col='path',
    y_col='score',
    target_size=(longitud, altura),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Visualización de las cinco primeras imágenes del generador de entrenamiento
fig, ax = plt.subplots(1, 5, figsize=(20, 4))
for i in range(5):
    batch = next(train_generator)
    image = batch[0][0]
    image = np.clip(image, 0, 1)
    ax[i].imshow(image)
    ax[i].axis('off')
plt.show()

# Calcular los pesos de las clases
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_data['score']), y=train_data['score'])
class_weight_dict = dict(zip(np.unique(train_data['score']), class_weights))

# Crear el modelo base EfficientNet B0 desde TensorFlow Hub
input_shape = (224, 224, 3)
img_input = layers.Input(shape=input_shape)
base_model = hub.KerasLayer('https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1', trainable=True)(img_input)
x = layers.Dense(256, activation='relu')(base_model)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)
model = Model(inputs=img_input, outputs=predictions)

# Compilar el modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', Recall(), Precision(), AUC()]
)

# Callbacks para el entrenamiento
checkpoint = ModelCheckpoint(
    "content/models/retinas.keras",
    monitor="val_accuracy",
    verbose=2,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch"
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.01,
    patience=2,
    verbose=1,
    mode='auto',
    min_lr=0.00001
)

early_stopping = EarlyStopping(
    monitor='val_recall',
    mode='max',
    verbose=1,
    patience=2,
    restore_best_weights=True
)

# Convertir los pesos de clase en un diccionario con claves enteras
class_weight_dict = {int(k): v for k, v in class_weight_dict.items()}

# Entrenamiento del modelo
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_data) // batch_size,
    epochs=epocas,
    validation_data=val_generator,
    validation_steps=len(val_data) // batch_size,
    callbacks=[checkpoint, reduce_lr, early_stopping],
    class_weight=class_weight_dict
)

# Guardar el modelo entrenado y los pesos
model.save('content/densenet121_retinopathy_model.h5')
model.save_weights('content/densenet121_retinopathy_weights.h5')

# Evaluar el modelo en el conjunto de prueba
eval = model.evaluate(test_generator, steps=len(test_data) // batch_size)
print(f"Evaluación del modelo - Pérdida: {eval[0]}, Precisión: {eval[1]}")

# Visualizar las curvas de entrenamiento
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
recall = history.history['recall']
val_recall = history.history['val_recall']
precision = history.history['precision']
val_precision = history.history['val_precision']
auc = history.history['auc']
val_auc = history.history['val_auc']
epochs_range = range(epocas)

plt.figure(figsize=(16, 8))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.subplot(2, 2, 3)
plt.plot(epochs_range, recall, label='Training Recall')
plt.plot(epochs_range, val_recall, label='Validation Recall')
plt.legend(loc='lower right')
plt.title('Training and Validation Recall')

plt.subplot(2, 2, 4)
plt.plot(epochs_range, auc, label='Training AUC')
plt.plot(epochs_range, val_auc, label='Validation AUC')
plt.legend(loc='lower right')
plt.title('Training and Validation AUC')

plt.show()

# Generar predicciones para el conjunto de prueba
test_generator.reset()
predictions = model.predict(test_generator, steps=len(test_data) // batch_size + 1)
predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]

# Obtener los nombres de los archivos en el conjunto de prueba
filenames = test_generator.filenames
results = pd.DataFrame({
    'Filename': filenames,
    'Prediction': predicted_classes
})

# Guardar las predicciones en un archivo CSV
results.to_csv('predicciones.csv', index=False)
print("Predicciones guardadas en predicciones.csv")
