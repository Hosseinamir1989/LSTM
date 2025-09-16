from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_lstm_model(window_size, n_features, learning_rate):
    model = Sequential([
        LSTM(64, input_shape=(window_size, n_features),return_sequences=True),
        LSTM(128),
        #Dropout(0.2),

        #Dense(128, activation='relu'),
        Dense(64, activation='relu'),
    
        Dense(1),
        
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model
