# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # TCC Strictu Sensu - Algoritmo de Classificação Binária: Melanoma versus Nevus.
# %% [markdown]
# ## 1. Definição do Problema
# 
# *Por que?*
# Câncer de pele é um dos cânceres mais comuns na atualidade, se identificado no início, pode ser mais facilmente tratado. A conscientização é muito importante, devemos informar a população de modo geral e facilitar o teste, como por exemplo um simples carregar de uma foto em um aplicativo que já te retornará as chances de ser uma pinta ou mancha na pele se desenvolver em um câncer de pele.
#  
# *Quem?*
# Este projeto visa ajudar todos que tiverem acesso a internet, ou smartphones, provendo uma maneira fácil de avaliar se pintas ou manchas na pele podem ser um câncer de pele em estágio inicial
#  
# *O que?*
# Um modelo de machine learning(mais precisamente deep learninng) que permitirá que usuários externos testem pintas ou manchas, buscando por possíveis câncer de pele.
#  
# *Quando?*
# O modelo deverá responder de maneira instantânea, ou mais próxima ao tempo real, como por exemplo entre 1 a 5 minutos.
#  
# *Onde?*
# A princípio através de input manual nesse projeto, mas no futuro usuários poderão carregar e testar imagens por um site ou aplicativo de Smartphone.
# 
# %% [markdown]
# ## 2. Coleta dos Dados
# 
# Segundo os requisitos deste projeto, duas bases de dados são requeridas. O tema deste projeto tem sido muito trabalhado por vários projetos ao longo dos últimos anos, porém existem apenas algumas bases muito populares, praticamente todas providas pela mesma fonte a ISIC que anualmente (desde 2016) tem lançado um desafio de Machine Learning e providenciado datasets com milhares de imagens com este tema. Entretanto como exigência do projeto tive de buscar uma base de dados distinta e após pesquisa encontrei um dataset pequeno mas que me fornece imagens de Nevus normais e melanoma. A fim de unir as bases precisei reduzir o escopo do dataset provido pela HAM10000 (Imagens colhidas pela ISIC em 2018), em apenas Nevus normais ou melanomas.

# %%
import pandas as pd
import shutil
import os


# %%
FINAL_PATH = "dataset_final//imagens//"


# %%
path_HAM10000 = "dataset_1//HAM10000_images//"
df_HAM10000_csv = "dataset_1//HAM10000_metadata.csv"

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
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# %% [markdown]
# !cd dataset_final/imagens/ && ls

# %%
imagens = []
for arquivo in os.listdir(FINAL_PATH):
  img = cv2.imread(os.path.join(FINAL_PATH,arquivo), 1)
  if img is not None:
    imagens.append(img)


# %%
print('Quantidade de Imagens: ', len(imagens))


# %%
altura=[]
largura=[]
canal=[]
  
for i in range(7983,7988):
  print("Imagem:",i+1)
  x,y,z=imagens[i].shape
  largura.append(x)
  altura.append(y)
  canal.append(z)
  print('largura: ' + str(x) + ', altura: ' + str(y) + ', canais: '+ str(z))
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


# %%
get_ipython().system('pwd')


# %%
pele_df = pd.read_csv('dataset_final/classificacao_dataset.csv')
print(pele_df.head())
np.random.seed(42)

SIZE=64


# %%
pele_df_img = pele_df.copy()


# %%
image_path = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('dataset_final', '*', '*.jpg'))}
#Define the path and add as a new column
pele_df_img['path'] = pele_df['image_id'].map(image_path.get)
#Use the path to read images.
pele_df_img['imagem'] = pele_df_img['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE,SIZE))))

pele_df_img.head()


# %%
num_exemplos = 3  

# Plot
fig, m_axs = plt.subplots(2, num_exemplos, figsize = (4*num_exemplos, 2*3))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         pele_df_img.sort_values(['type']).groupby('type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(num_exemplos, random_state=1234).iterrows()):
        c_ax.imshow(c_row['imagem'])
        c_ax.axis('off')

# %% [markdown]
# ## 5. Criação do Modelo de ML
# %% [markdown]
# ### Testando Diversos Modelos

# %%
import os 
import pandas as pd
import numpy as np
import tensorflow as tf 
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# %%
#Setting Global Variables
FINAL_PATH = "dataset_final//imagens//"

TRAIN_PATH = "dataset_final//train//"

TEST_PATH = "dataset_final//test//"

SIZE=224


# %%
pele_df = pd.read_csv('dataset_final/classificacao_dataset.csv')


le = LabelEncoder()
le.fit(pele_df['type'])
LabelEncoder()
print('label 0 é igual á ' + list(le.classes_)[0])
print('label 1 é igual á ' + list(le.classes_)[1])

pele_df['label'] = le.transform(pele_df["type"]) 

print(pele_df['label'].value_counts())

print(pele_df.head())


# %%
pele_df_final = pele_df.copy()

image_path = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('dataset_final', '*', '*.jpg'))}

#Definindo o diretório como uma nova coluna
pele_df_final['path'] = pele_df['image_id'].map(image_path.get)

#Usar o dataframe completo estava levando muito tempo, portanto tive de quebra-lo num fração menor, ainda com mais de 2000 linhas
pele_df_final_frac = pele_df_final.sample(frac=0.35)

print(pele_df_final_frac['type'].value_counts())

X = pele_df_final_frac['path']
Y = pele_df_final_frac['type'] #Assign label values to Y


# %%
print(f'Valores únicos para Y: {list(set(Y))}')

x_train_auto, x_test_auto, y_train_auto, y_test_auto = train_test_split(X, Y, test_size=0.20, random_state=42)

print(f'x_train_auto: {x_train_auto[0:5]}')
print(f'x_test_auto: {x_test_auto[0:5]}')
print(f'y_train_auto: {y_train_auto[0:5]}')
print(f'y_test_auto: {y_test_auto[0:5]}')


# %%
df_train = pd.DataFrame(columns=['image_path','type'])
df_train['image_path'] = x_train_auto
df_train['type'] = y_train_auto

df_test = pd.DataFrame(columns=['image_path','type'])
df_test['image_path'] = x_test_auto
df_test['type'] = y_test_auto


# %%
#Creating a directory for Train 
for n, row in df_train.iterrows():
    if row['type']=='mel':
        shutil.copy2(row['image_path'], TRAIN_PATH + 'mel//')
    else: 
        shutil.copy2(row['image_path'], TRAIN_PATH+ 'nv//')


# %%
#Creating a directory for Test
for n, row in df_test.iterrows():
    if row['type']=='mel':
        shutil.copy2(row['image_path'], TEST_PATH + 'mel//')
    else: 
        shutil.copy2(row['image_path'], TEST_PATH+ 'nv//')


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
        f.savefig('./imagens_resultados/' + str(savename))

# %% [markdown]
# ### Testing VGG-16

# %%
SIZE = 224


# %%
# Image Augmentation

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )


# %%
# Training and Validation Sets
# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(TRAIN_PATH, batch_size = 20, class_mode = 'binary', target_size = (SIZE, SIZE))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(TEST_PATH,  batch_size = 20, class_mode = 'binary', target_size = (SIZE, SIZE))


# %%
# Loading the Base Model
from keras.applications.vgg16 import VGG16

base_model = VGG16(input_shape = (SIZE, SIZE, 3), # Shape of our images
include_top = True, # Leave out the last fully connected layer
weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False


# %%
# Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])


# %%
vgghist = model.fit(train_generator, validation_data = validation_generator, validation_steps= 8, batch_size=20, steps_per_epoch = 20, epochs = 8)

# %% [markdown]
# ## Testing Inception

# %%
SIZE = 150


# %%
# Data Augmentation
# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, width_shift_range = 0.2,
 height_shift_range = 0.2,shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator( rescale = 1.0/255. )


# %%
# Training and Validation Generators

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(TRAIN_PATH, batch_size = 20, class_mode = 'binary', target_size = (SIZE, SIZE))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(TEST_PATH,  batch_size = 20, class_mode = 'binary', target_size = (SIZE, SIZE))


# %%
# Loading the Base Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

base_model = InceptionV3(input_shape = (SIZE, SIZE, 3), include_top = False, weights = 'imagenet')


# %%
# Compile and Fit

for layer in base_model.layers:
    layer.trainable = False

from tensorflow.keras.optimizers import RMSprop

x = layers.Flatten()(base_model.output)
x = layers.Dense(18432, activation='relu')(x)
x = layers.Dropout(0.2)(x)

# Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer = RMSprop(learning_rate=0.0001), loss = 'binary_crossentropy', metrics = ['acc'])


# %%
inc_history = model.fit(train_generator, validation_data = validation_generator, validation_steps= 8,  batch_size=20, steps_per_epoch = 20, epochs = 8)

# %% [markdown]
# ## Testing ResNet50

# %%
SIZE=224


# %%
# Data Augmentation and Generators
# Add our data-augmentation parameters to ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, width_shift_range = 0.2,
 height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1.0/255.)


# %%

train_generator = train_datagen.flow_from_directory(TRAIN_PATH, batch_size = 20, class_mode = 'binary', target_size = (SIZE, SIZE))

validation_generator = test_datagen.flow_from_directory(TEST_PATH, batch_size = 20, class_mode = 'binary', target_size = (SIZE, SIZE))


# %%
#Import the base model
from tensorflow.keras.applications import ResNet50

base_model = ResNet50(input_shape=(SIZE, SIZE,3), include_top=False, weights="imagenet")

for layer in base_model.layers:
    layer.trainable = False


# %%
# Build and Compile the Model
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten

base_model = Sequential()
base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max'))
base_model.add(Dense(1, activation='sigmoid'))

#opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.0001)
base_model.compile(optimizer = 'SGD', loss = 'binary_crossentropy', metrics = ['acc'])


# %%
resnet_history = base_model.fit(train_generator, batch_size=20, validation_data = validation_generator, validation_steps= 8, 
steps_per_epoch = 20, epochs = 8)

# %% [markdown]
# ### CNN com Tensorflow

# %%
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


# %%
SIZE = 224


# %%
# Data Augmentation and Generators
# Add our data-augmentation parameters to ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1.0/255.)


# %%
train_generator = train_datagen.flow_from_directory(TRAIN_PATH, batch_size = 20, class_mode = 'binary', target_size = (SIZE, SIZE))

validation_generator = test_datagen.flow_from_directory(TEST_PATH, batch_size = 20, class_mode = 'binary', target_size = (SIZE, SIZE))


# %%
model = tf.keras.models.Sequential([
# Note the input shape is the desired size of the image 200x200 with 3 bytes color
# This is the first convolution
tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(SIZE, SIZE, 3)),
tf.keras.layers.MaxPooling2D(2, 2),
# The second convolution
tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
# The third convolution
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
# The fourth convolution
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
# # The fifth convolution
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
# Flatten the results to feed into a DNN
tf.keras.layers.Flatten(),
# 512 neuron hidden layer
tf.keras.layers.Dense(512, activation='relu'),
# Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('dandelions') and 1 for the other ('grass')
tf.keras.layers.Dense(1, activation='sigmoid')
])


# %%
from tensorflow.keras.optimizers import RMSprop
model.compile(loss='binary_crossentropy',
optimizer=RMSprop(learning_rate=0.001),
metrics='accuracy')


# %%
cnn_history = model.fit(train_generator, steps_per_epoch=20, epochs=8, verbose=1, validation_data = validation_generator, validation_steps=8)

# %% [markdown]
# ## 6. Interpretação dos Resultados
# %% [markdown]
# Para análise do modelo, utilizei dos seguintes conceitos: 
# 
# * Coeficiente de Perda
# * Acurácia
# * Curva ROC
# %% [markdown]
# ### VGG16

# %%
#VGG16
model.evaluate(validation_generator)


# %%
plt.plot(vgghist.history['acc'])
plt.plot(vgghist.history['val_acc'])
plt.title('Model accuracy - VGG16')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# %%
STEP_SIZE_TEST=validation_generator.n//validation_generator.batch_size
validation_generator.reset()
preds = model.predict(validation_generator,
verbose=1)


# %%
fpr, tpr, _ = roc_curve(validation_generator.classes, preds)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('VGG16')
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# ### InceptionV3

# %%
# Inceptionv3
model.evaluate(validation_generator)


# %%
plt.plot(inc_history.history['acc'])
plt.plot(inc_history.history['val_acc'])
plt.title('Model accuracy - Inception')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# %%
STEP_SIZE_TEST=validation_generator.n//validation_generator.batch_size
validation_generator.reset()
preds = model.predict(validation_generator,
verbose=1)


# %%
fpr, tpr, _ = roc_curve(validation_generator.classes, preds)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Inception')
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# ### ResNet50

# %%
# ResNet50
model.evaluate(validation_generator)


# %%
plt.plot(resnet_history.history['acc'])
plt.plot(resnet_history.history['val_acc'])
plt.title('Model accuracy - ResNet50')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# %%
STEP_SIZE_TEST=validation_generator.n//validation_generator.batch_size
validation_generator.reset()
preds = model.predict(validation_generator,
verbose=1)


# %%
fpr, tpr, _ = roc_curve(validation_generator.classes, preds)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ResNet50')
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# ### CNN Manualmente Composto

# %%
model.evaluate(validation_generator)


# %%
STEP_SIZE_TEST=validation_generator.n//validation_generator.batch_size
validation_generator.reset()
preds = model.predict(validation_generator,
verbose=1)


# %%
fpr, tpr, _ = roc_curve(validation_generator.classes, preds)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('CNN Manualmente Composto')
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# ## 7. Comunicação dos Resultados

# %%
get_ipython().system('pwd')

# %% [markdown]
# Por favor visite o Documento: 
# 
# https://github.com/cesaraugusto98/tcc_pos/blob/main/imagens_resultados/TCC-workflow-canvas.png
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
# 
# Top 4 Pre-Trained Models for Image Classification with Python Code | https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/
# 
# Binary Image Classification | https://www.analyticsvidhya.com/blog/2021/06/binary-image-classifier-using-pytorch/
# 
# CNN Binary Image Classifier in TensorFlow | https://towardsdatascience.com/10-minutes-to-building-a-cnn-binary-image-classifier-in-tensorflow-4e216b2034aa
# 
# Different Types od CNN Models | https://iq.opengenus.org/different-types-of-cnn-models/
# 
# A Data Science Workflow Canvas to Kickstart Your Projects | https://towardsdatascience.com/a-data-science-workflow-canvas-to-kickstart-your-projects-db62556be4d0