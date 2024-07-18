import numpy as np
from numpy import loadtxt
from keras.models import model_from_json

# Load dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
x = dataset[:, 0:8]
y = dataset[:, 8]

# Load model from JSON file
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load weights into the model
model.load_weights("model.h5")
print("Loaded model from disk")

# Make predictions
predictions = model.predict(x)
predicted_classes = np.argmax(predictions, axis=1)

# Print some predictions
for i in range(5, 10):
    print('%s => %d (expected %d)' % (x[i].tolist(), predicted_classes[i], y[i]))
