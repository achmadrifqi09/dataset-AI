from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

import cv2
import numpy as np
import glob

img_dir = "dataset/images/"
# img_dir2 = "dataset/Kemangi/"

label_list = ['Kemangi', 'Pepaya']
data = []
labels = []

# ext = ['jpg', 'png']

for label in label_list :
    for imgPath in glob.glob(img_dir + label + '\\*.jpg'):
       image = cv2.imread(imgPath)
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
model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(1024, activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.summary()

lr = 0.01
max_epochs = 100
opt_funct = SGD(learning_rate=lr)

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


i = q_pred.argmax(axis=1)[0]
label = lb.classes_[i]

text = "{}: {:.2f}%".format(label, q_pred[0][i] *100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow('Output', output)
cv2.waitKey() # image tidak akan diclose,sebelum user menekan sembarang tombol
cv2.destroyWindow('Output') # image akan diclose