# core/autoencoder_compressor.py
import numpy as np
import cv2
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout
import os

class AutoencoderCompressor:
    def __init__(self, model_path=None, input_shape=(256, 256, 3)):
        self.input_shape = input_shape
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
            print(f"Loaded pre-trained model from {model_path}")
        else:
            self.model = self._build_advanced_model()
            print("Initialized new autoencoder model")

    def _build_advanced_model(self):
        input_img = Input(shape=self.input_shape)
        
        # Encoder
        x = Conv2D(64, (3,3), activation='relu', padding='same')(input_img)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2,2), padding='same')(x)
        x = Dropout(0.25)(x)
        
        x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2,2), padding='same')(x)
        x = Dropout(0.25)(x)
        
        x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        encoded = MaxPooling2D((2,2), padding='same')(x)
        
        # Decoder
        x = Conv2D(256, (3,3), activation='relu', padding='same')(encoded)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
        x = Dropout(0.25)(x)
        
        x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
        x = Dropout(0.25)(x)
        
        x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
        
        decoded = Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)
        
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def compress(self, image, output_path):
        # המרה לצבע אם צריך
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        # שמירת גודל מקורי
        original_size = image.nbytes
        
        # עיבוד תמונה
        img_resized = cv2.resize(image, self.input_shape[:2])
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # דחיסה ושחזור
        reconstructed = self.model.predict(img_batch, verbose=0)
        compressed_img = (reconstructed[0] * 255).astype(np.uint8)
        
        # שמירת התמונה הדחוסה
        cv2.imwrite(output_path, compressed_img)
        compressed_size = os.path.getsize(output_path)
        
        # חישוב יחס דחיסה
        compression_ratio = compressed_size / original_size if original_size > 0 else 0
        
        return compression_ratio