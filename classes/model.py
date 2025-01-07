import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Masking
from tensorflow.keras.callbacks import EarlyStopping

class Model:
    def __init__(
            self, 
            n_features=3, 
            n_classes=5, 
            padding_value=-999, 
            seq_len=59, 
            load=False, 
            path=None, 
            padding='post',
            layer_type='lstm',  # lstm, gru, bi_lstm
            layer_size=128,
            epochs=50,
            batch_size=64,
            x_train=None,
            y_train=None,
            x_val=None,
            y_val=None,
            debug=False
        ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.padding_value = padding_value
        self.seq_len = seq_len
        self.load = load
        self.path = path
        self.padding = padding
        self.layer_type = layer_type
        self.layer_size = layer_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.debug = debug
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.history = None
        self.y_pred_unthreshed = []

        if self.load:
            self._load_model()
        else:
            self._train_model()


    def evaluate_native(self, x, y):
        self.model.evaluate(x, y)

    
    def evaluate(self, x, y, compute_preds=False):
        if compute_preds:
            self.predict(x)

        acc = (self.y_pred == y).all(axis=1).sum() / len(y)

        return acc


    def get_model(self):
        return self.model
        

    def _load_model(self):
        self.model = tf.keras.models.load_model(self.path)

    
    def _train_model(self):
        model = self._get_compiled()
        self.model, self.history = self._fit(model)

        if self.debug:
            self.visualize_history()


    def visualize_history(self):
        if self.history is None:
            print("The model was loaded from memory or history has corrupted.")
            return
        
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.show()

        plt.plot(self.history.history['accuracy'], label='Accuracy')
        plt.legend()
        plt.show()

    
    def save(self, path):
        self.model.save(path)


    def _fit(self, model, callbacks=['early']):
        callbacks = []

        if 'early' in callbacks:
            early = EarlyStopping(patience=10, restore_best_weights=True)
            callbacks.append(early)
        
        history = model.fit(
            self.x_train, self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.x_val, self.y_val),
            callbacks=callbacks
        )

        return model, history


    def _get_compiled(self):
        model = Sequential()
        model.add(Masking(mask_value=self.padding_value, input_shape=(self.seq_len, self.n_features)))

        if self.padding == 'post':  # pre padding is not supported by cudnn
            if self.layer_type == 'lstm':
                model.add(LSTM(self.layer_size, return_sequences=False))
            elif self.layer_type == 'gru':
                model.add(GRU(self.layer_size, return_sequences=False))
            else:
                model.add(Bidirectional(LSTM(self.layer_size, return_sequences=False)))
        else:
            model.add(LSTM(self.layer_size, return_sequences=False, use_cudnn=False))

        model.add(Dense(self.n_classes, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', metrics=['accuracy'])

        if self.debug:
            print(model.summary())

        return model
    

    def _apply_threshold(self, x, thresh=0.5):
        arr = x.copy()

        arr[arr > thresh] = 1
        arr[arr <= thresh] = 0
        
        return arr

    
    def predict(self, x, thresh=0.5):
        self.y_pred_unthreshed = self.model.predict(x)
        self.y_pred = self._apply_threshold(self.y_pred_unthreshed, thresh)

        return self.y_pred