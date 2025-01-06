import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(
            self, 
            padding_value=-999, 
            padding='post', 
            train_size=0.8, 
            test_size=0.1, 
            val_size=0.1, 
            random_state=42, 
            debug=False
        ):
        self.padding_value = padding_value
        self.padding = padding
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.debug = debug
        
    
    def get_data(self, xl, yl, masks, return_indices=False):
        X, y = self._get_preprocessed(xl, yl)
        train_indices, test_indices, val_indices = self._get_indices(X)

        x_train, x_test, x_val = self._split_data(X, train_indices, test_indices, val_indices)
        y_train, y_test, y_val = self._split_data(y, train_indices, test_indices, val_indices)
        masks_train, masks_test, masks_val = self._split_data(masks, train_indices, test_indices, val_indices, np_encode=False)

        if self.debug:
            print(f"Train / Test / Val: {len(x_train)} / {len(x_test)} / {len(x_val)}")

        if return_indices:
            return (x_train, y_train, masks_train, x_test, y_test, masks_test, x_val, y_val, masks_val, 
                    train_indices, test_indices, val_indices)

        return x_train, y_train, masks_train, x_test, y_test, masks_test, x_val, y_val, masks_val
    

    def _split_data(self, x, train_indices, test_indices, val_indices, np_encode=True):
        if np_encode:
            x_train = np.array([x[i] for i in train_indices])
            x_test = np.array([x[i] for i in test_indices])
            x_val = np.array([x[i] for i in val_indices])

            return x_train, x_test, x_val
        
        return [x[i] for i in train_indices], [x[i] for i in test_indices], [x[i] for i in val_indices]
    

    def _get_indices(self, X):
        train_indices, temp_indices = train_test_split(
            np.arange(len(X)), 
            test_size=(self.test_size + self.val_size), 
            random_state=self.random_state
        )
        test_indices, val_indices = train_test_split(
            temp_indices, 
            test_size=self.test_size / (self.test_size + self.val_size), 
            random_state=self.random_state
        )
        
        return train_indices, test_indices, val_indices


    def _get_preprocessed(self, xl, yl):
        seq_len = self._get_seq_len(xl)
        X = pad_sequences(xl, maxlen=seq_len, dtype='float32', padding=self.padding, value=self.padding_value)
        y = np.array(yl)

        if self.debug:
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")

        return X, y


    def _get_seq_len(self, xl):
        return max(len(seq) for seq in xl)
    

    def _normalize(self, X):
        return (X - X.mean(axis=0)) / X.std(axis=0)