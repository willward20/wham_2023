import tensorflow as tf


converter = tf.lite.TFLiteConverter.from_saved_model('model_04081025')
model = converter.convert()

open('model_final.tflite', 'wb').write(model)
