from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

img_dir = "dataset/images/"
# img_dir2 = "dataset/Kemangi/"

label_list = ['Kemangi', 'Pepaya']
data = []
labels = []

# ext = ['jpg', 'png']

for label in label_list:
    for imagePath in glob.glob(img_dir+label+'\\*.jpg'):
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (32, 32))
        data.append(image)
        labels.append(label)


np.array(data).shape


data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)


print(labels)

lb = LabelEncoder()
labels = lb.fit_transform(labels)
print(labels)


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


print('Ukuran data train =', x_train.shape)
print('Ukuran data test =', x_test.shape)


model = Sequential()

model.add(InputLayer(input_shape=[32,32,3]))
model.add(Conv2D(filters=32, kernel_size=2, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, padding='same'))
model.add(Conv2D(filters=50, kernel_size=2, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

lr = 0.001
max_epochs = 100
opt_funct = Adam(learning_rate=lr)

model.compile(loss = 'binary_crossentropy', 
              optimizer = opt_funct, 
              metrics = ['accuracy'])


H = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=max_epochs, batch_size=32)

N = np.arange(0, max_epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.xlabel("Epoch #")
plt.legend()
plt.show()


predictions = model.predict(x_test, batch_size=32)
target = (predictions > 0.5).astype(np.int)
print(classification_report(y_test, target, target_names=label_list))

queryPath = img_dir+'pepaya_test.jpg'
query = cv2.imread(queryPath)
output = query.copy()
query = cv2.resize(query, (32, 32))
q = []
q.append(query)
q = np.array(q, dtype='float') / 255.0

q_pred = model.predict(q)
print(q_pred)


if q_pred<=0.5 :
    target = "Daun Kemangi"
else :
    target = "Daun Pepaya"
text = "{}".format(target)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow('Output', output)
cv2.waitKey() 
cv2.destroyWindow('Output')

model.save('model-save.h5')