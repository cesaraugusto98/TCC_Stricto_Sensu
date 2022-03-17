# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # TCC Strictu Sensu - Skin Cancer detector
# %% [markdown]
# ## 1. Definição do Problema
# *Porque?* 
# Cancer de pele é um dos cancer mais comuns na atualidade, se identificado no inicio, pode ser mais fácilmente tratado. A concientização é muito importante, devemos informar a população de modo geral e facilitar o teste, como por exemplo um simples carregar de uma foto em um aplicativo que já te retornará as chances de ser uma pinta ou mancha na pele se desenvolver em um cancer de pele.
# 
# *Quem?* 
# Este projeto visa ajudar todos que tiverem acesso a internet, ou smartphones, provendo uma maneira fácil de avaliar se pintas ou manchas na pele podem ser um cancer de pele em estágio inicial
# 
# *Oque?*
# Um modelo de machine learning(mais precisamente deep learninng) que permitará que usuários externos testem pintas ou manchas, buscando por possiveis cancer de pele.
# 
# *Quando?*
# O modelo deverá responder de maneira instantanea, ou mais próxima ao tempo real como 1 a 5 minutos. 
# 
# *Onde?*
# A principio através de input manual nesse projeto, mas no futuro usuários poderão carregar e testar imagens por um site ou applicativo de Smartphone.

# %%


# %% [markdown]
# ## 2. Coleta dos Dados
# Segundo os requisitos desse projeto, duas bases de dados são requeridas. O tema desse projeto tem sido muito trabalhado por vários projetos ao longo dos últimos anos, porém existem apenas algumas bases muito populares, praticamente todas providas pela mesma fonte a ISIC que anualmente (desde 2016) tem lançado um desafio de Machine Learning e providenciado datasets com milhares de imagens com este tema. Entretando como exigência do projeto tive de buscar uma base de dados distinta e após pesquisa encontrei um dataset pequeno mas que me fornece imagens de Nevuas normais e melanoma. Afim de unir as bases precisei reduzir o escopo do dataset provido pela HAM10000 (Imagens colhidas pela ISIC em 2018), em apenas Nevuas normais ou melanomas.

# %%
import pandas as pd
import shutil
import os


# %%
FINAL_PATH = "dataset_final//imagens//"


# %%
path_HAM10000 = "dataset_1//HAM10000_images//"
df_HAM10000_csv = "dataset_1//HAM10000_metadata"

df_HAM10000_csv = pd.read_csv(df_HAM10000_csv)

#Tamanho original
print('Tamanho Original: ', df_HAM10000_csv.shape)

#Filtrando apenas melanocytic nevi (begnigno) e melanoma(maligno), 
#o resto da base será desconsiderado para este estudo.
df_HAM10000_csv_filtered=df_HAM10000_csv[(df_HAM10000_csv.dx == "nv") | (df_HAM10000_csv.dx == "mel")]

#Tamanho após filtrado
print('Tamanho após aplifcar o filtro: ', df_HAM10000_csv_filtered.shape)

#Movendo os arquivos para Pasta Final
for index, row in df_HAM10000_csv_filtered.iterrows(): 
    shutil.copy2(path_HAM10000 + row['image_id'] + '.jpg', FINAL_PATH)


# %%
df_mednode = pd.DataFrame(columns=['image_id','type'])
path_mednode = "dataset_2//complete_mednode_dataset//"

for diretorio,subpasta, arquivos in os.walk(path_mednode):
    if diretorio !="dataset_2//complete_mednode_dataset//":
        for arquivo in arquivos:
            if 'melanoma' in diretorio:
                df_mednode=df_mednode.append({'image_id': arquivo.replace('.jpg',''), 'type':'mel'}, ignore_index=True)
                shutil.copy2(diretorio +'//'+ arquivo, FINAL_PATH)
            elif 'naevus' in diretorio:
                df_mednode=df_mednode.append({'image_id': arquivo.replace('.jpg',''), 'type':'nv'}, ignore_index=True)
                shutil.copy2(diretorio +'//'+ arquivo, FINAL_PATH)
        
print(df_mednode.shape)
df_mednode.head()


# %%
#Unindo os dataframes de Typagem de cada imagem:

print('Dataframe HAM10000: ', df_HAM10000_csv_filtered.shape)
print('Dataframe HAM10000 colunas: ', df_HAM10000_csv_filtered.columns)

print('Dataframe Med Node: ', df_mednode.shape)
print('Dataframe Med Node colunas: ', df_mednode.columns)

df_HAM10000_final=df_HAM10000_csv_filtered.drop(columns=['lesion_id', 'dx_type', 'age', 'sex', 'localization', 'dataset'])
df_HAM10000_final=df_HAM10000_final.rename(columns = {'image_id': 'image_id', 'dx': 'type'}, inplace = False)

df_final = pd.DataFrame(columns=['image_id','type'])
df_final = df_final.append(df_HAM10000_final, ignore_index=True)
df_final = df_final.append(df_mednode, ignore_index=True)

print('Dataframe final: ' , df_final.shape)

print(df_final.head())
print(df_final.tail())

df_final.to_csv("dataset_final//" + "classificacao_dataset.csv")

# %% [markdown]
# ## 3. Processamento/Tratamento dos Dados
# 

# %%
import albumentations as alb
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


# %%
get_ipython().system('cd dataset_final/imagens/ && ls')


# %%
imagens = []
for arquivo in os.listdir(FINAL_PATH):
  img = cv2.imread(os.path.join(FINAL_PATH,arquivo))
  if img is not None:
    imagens.append(img)


# %%
print('Quantidade de Imagens: ', len(imagens))


# %%
altura=[]
largura=[]
canal=[]
  
for i in range(2):
  print("Imagem:",i+1)
  x,y,z=imagens[i].shape
  largura.append(x)
  altura.append(y)
  canal.append(z)
  print('largura: ' + str(x) + ', altura: ' + str(y) + ', canal: '+ str(z))
  plt.imshow(imagens[i])
  plt.show()

# %% [markdown]
# ## 4. Análise/Exploração dos Dados
# 
# 

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf


# %%
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import autokeras as ak


# %%
get_ipython().system('pwd')


# %%
pele_df = pd.read_csv('dataset_final/classificacao_dataset.csv')
print(pele_df.head())
np.random.seed(42)

SIZE=32


# %%
# label encoding to numeric values from text
le = LabelEncoder()
le.fit(pele_df['type'])
LabelEncoder()
print(list(le.classes_))
print('label 0 é igual á ' + list(le.classes_)[0])
print('label 1 é igual á ' + list(le.classes_)[1])

pele_df['label'] = le.transform(pele_df["type"]) 
print(pele_df.sample(10))


# %%
# Data distribution visualization
fig = plt.figure(figsize=(15,10))

ax = fig.add_subplot(221)
pele_df['type'].value_counts().plot(kind='bar', ax=ax)
ax.set_ylabel('Quantidade')
ax.set_title('Tipo de lesão');
ax.bar_label(ax.containers[0])

plt.tight_layout()
plt.show()

print(pele_df['label'].value_counts())


# %%
new_pele_df = pele_df.copy()


# %%
#Distribution of data into various classes 
from sklearn.utils import resample


#Balance data.
# Many ways to balance data... you can also try assigning weights during model.fit
#Separate each classes, resample, and combine back into single dataframe

df_0 = pele_df[pele_df['label'] == 0]
df_1 = pele_df[pele_df['label'] == 1]


n_samples=1500
df_mel_balanceado = resample(df_0, replace=True, n_samples=n_samples, random_state=42) 
df_nv_balanceado = resample(df_1, replace=True, n_samples=n_samples, random_state=42)

pele_df_balanceado = pd.concat([df_mel_balanceado, df_nv_balanceado])


# %%
#pele_df_balanceado
#Now time to read images based on image ID from the CSV file
#This is the safest way to read images as it ensures the right image is read for the right ID
print(pele_df_balanceado['label'].value_counts())

image_path = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('dataset_final', '*', '*.jpg'))}
#Define the path and add as a new column
pele_df_balanceado['path'] = pele_df['image_id'].map(image_path.get)
#Use the path to read images.
pele_df_balanceado['imagem'] = pele_df_balanceado['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE,SIZE))))

pele_df_balanceado.head()


# %%
#pele_df_não_balanceado
#Now time to read images based on image ID from the CSV file
#This is the safest way to read images as it ensures the right image is read for the right ID
print(new_pele_df['label'].value_counts())

pele_df_nao_balanceado = new_pele_df.copy()

image_path = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('dataset_final', '*', '*.jpg'))}
#Define the path and add as a new column
pele_df_nao_balanceado['path'] = new_pele_df['image_id'].map(image_path.get)
#Use the path to read images.
pele_df_nao_balanceado['imagem'] = pele_df_nao_balanceado['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE,SIZE))))

pele_df_nao_balanceado.head()


# %%
num_exemplos = 3  

# Plot
fig, m_axs = plt.subplots(2, num_exemplos, figsize = (4*num_exemplos, 2*3))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         pele_df_balanceado.sort_values(['type']).groupby('type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(num_exemplos, random_state=1234).iterrows()):
        c_ax.imshow(c_row['imagem'])
        c_ax.axis('off')

fig, m_axs = plt.subplots(2, num_exemplos, figsize = (4*num_exemplos, 2*3))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         pele_df_nao_balanceado.sort_values(['type']).groupby('type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(num_exemplos, random_state=1234).iterrows()):
        c_ax.imshow(c_row['imagem'])
        c_ax.axis('off')


# %%
#Convert dataframe column of images into numpy array
X_balanceado = np.asarray(pele_df_balanceado['imagem'].tolist())
X_balanceado = X_balanceado/255. # Scale values to 0-1. You can also used standardscaler or other scaling methods.
Y_balanceado = pele_df_balanceado['label'] #Assign label values to Y
Y_cat_balanceado = to_categorical(Y_balanceado, num_classes=2) #Convert to categorical as this is a multiclass classification problem
print('X_balanceado: ', X_balanceado)
print('Y_balanceado: ', Y_balanceado)


# %%
#Convert dataframe column of images into numpy array
X = np.asarray(pele_df_nao_balanceado['imagem'].tolist())
X = X /255. # Scale values to 0-1. You can also used standardscaler or other scaling methods.
Y = pele_df_nao_balanceado['label'] #Assign label values to Y
Y_cat = to_categorical(Y, num_classes=2) #Convert to categorical as this is a multiclass classification problem
print('X: ', X)
print('Y: ', Y)


# %%
#Split to training and testing. Get a very small dataset for training as we will be 
# fitting it to many potential models. 
x_train_auto_balanceado, x_test_auto_balanceado, y_train_auto_balanceado, y_test_auto_balanceado = train_test_split(X_balanceado, Y_cat_balanceado, test_size=0.80, random_state=42)


# %%
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


# %%
datagen.fit(x_train_auto_balanceado)

# %% [markdown]
# ## 5. Criação do Modelo de ML

# %%
model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(SIZE,SIZE,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2, activation="sigmoid"))

model.summary()


# %%
model.compile(
    loss=keras.losses.sparse_categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adadelta(),
    metrics=["accuracy"],
)


# %%
batch_size = 256
epochs = 50


# %%
#Modelo com base balanceada
history_balanceado  = model.fit(X_balanceado, Y_balanceado,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.2)


# %%
#Modelo com base não_balanceada
history_nao_balanceado = model.fit(X, Y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.2)

# %% [markdown]
# ## 6. Interpretação dos Resultados
# %% [markdown]
# ## Resultados base balanceada

# %%
# Plot training & validation accuracy values
plt.plot(history_balanceado.history['accuracy'])
plt.plot(history_balanceado.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history_balanceado.history['loss'])
plt.plot(history_balanceado.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# %%
model.evaluate(X_balanceado[-500:],Y_balanceado[-500:])


# %%
Y_pred_balanceado = model.predict(X_balanceado)

yd = Y_pred_balanceado[:, 1] - Y_pred_balanceado[:, 0]

most_mel = np.argsort(yd)
most_nv = np.argsort(yd)[::-1]
most_ambiguous = np.argsort(np.abs(yd))


# %%
plt.plot(np.sort(yd))


# %%
files = ["{}/{}".format(FINAL_PATH, fi) for fi in os.listdir(FINAL_PATH) if fi.endswith("jpg")]


# %%
def get_image_data(filename):
    img = Image.open(filename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data
    
def plot_N(indices, savename=None):    
    f, axarr = plt.subplots(len(indices) // 5, 5)
    f.set_size_inches(14, 14)
    f.subplots_adjust(wspace=0.2, hspace=0, left=0, right=1, top=0.4, bottom=0)
    for i in range(len(indices)):
        axarr[i // 5, i % 5].axis("off")
        axarr[i // 5, i % 5].imshow(get_image_data(files[indices[i]]))
    if savename is not None:
        f.savefig('./exported_images/' & savename)


# %%
plot_N(most_mel[:10], "most_mel.jpg")


# %%
plot_N(most_nv[:10], "most_nv.jpg")


# %%
plot_N(most_ambiguous[:30], "most_ambiguous.jpg")

# %% [markdown]
# ## Resultados Base não balanceada

# %%
# Plot training & validation accuracy values
plt.plot(history_nao_balanceado.history['accuracy'])
plt.plot(history_nao_balanceado.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history_nao_balanceado.history['loss'])
plt.plot(history_nao_balanceado.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# %%
model.evaluate(X[-500:],Y[-500:])


# %%
Y_pred = model.predict(X)

yd = Y_pred[:, 1] - Y_pred[:, 0]

most_mel = np.argsort(yd)
most_nv = np.argsort(yd)[::-1]
most_ambiguous = np.argsort(np.abs(yd))


# %%
plt.plot(np.sort(yd))

# %% [markdown]
# ## 7. Comunicação dos Resultados

# %%


# %% [markdown]
# ## 8. Referencias
# Tschandl, Philipp, 2018, "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions", https://doi.org/10.7910/DVN/DBW86T, Harvard Dataverse, V3, UNF:6:/APKSsDGVDhwPBWzsStU5A== [fileUNF]
# 
# I. Giotis, N. Molders, S. Land, M. Biehl, M.F. Jonkman and N. Petkov: "MED-NODE: A computer-assisted melanoma diagnosis system using non-dermoscopic images", Expert Systems with Applications, 42 (2015), 6578-6585 
# 
# Skin cancer detection: Applying a deep learning based model driven architecture in the cloud for classifying dermal cell images | https://www.sciencedirect.com/science/article/pii/S2352914819302047
# 
# PH2 dataset | https://www.fc.up.pt/addi/ph2%20database.html
# 
# 
# Aprendizagem Profunda Aplicada a Identificação de melanoma | https://tedebc.ufma.br/jspui/bitstream/tede/2578/2/LucasMaia.pdf
# 
# Github Project - Skin Cancer detection |
# https://github.com/Tirth27/Skin-Cancer-Classification-using-Deep-Learning
# 
# Good keras gudie | https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/
# 
# MACHINE LEARNING WITH PYTHON: TRAIN YOUR OWN IMAGE CLASSIFICATION MODEL WITH KERAS AND TENSORFLOW | https://mlconference.ai/blog/machine-learning-with-python/
# 
# Single Label Imagem Classification | https://blog.workaround.vercel.app/blog/single-label-image-classification-with-keras

# %%



