import tensorflow as tf


converter = tf.lite.TFLiteConverter.from_saved_model('model_04121906')
model = converter.convert()

open('model_04121906.tflite', 'wb').write(model)
