import tensorflow as tf

box_model = tf.keras.models.load_model('../models/best_lstm_model_40.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(box_model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
model = converter.convert()

model.summary()
# model.save('./new_mode.h5')
