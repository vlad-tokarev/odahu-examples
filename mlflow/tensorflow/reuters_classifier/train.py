
from __future__ import print_function

import mlflow.keras
import mlflow.pyfunc
from keras import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split


from src import parser
from src.utils import fit_tokenizer, fit_topics_encoder, ModelWrapper, MAX_WORDS, save_samples

# Constants
REUTERS_DATA_PATH = 'data'
TOPICS_ENCODER_FILE = 'topics_encoder.pkl'
TOKENIZER_FILE = 'tokenizer.pkl'
KERAS_MODEL_FILE = 'keras_model.h5'
BATCH_SIZE = 32
EPOCHS = 5

# Streamer of reuters dataset
streamer = parser.ReutersStreamer(REUTERS_DATA_PATH)

# Fitting transformers
tokenizer = fit_tokenizer(streamer)
topics_encoder = fit_topics_encoder(streamer)

# Prepare features (encode and split)
X = tokenizer.texts_to_matrix(
    (text for text, _ in streamer.stream_reuters_documents_with_topics())
)
Y = topics_encoder.transform(
    (topics for _, topics in streamer.stream_reuters_documents_with_topics())
)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Build model
print('Building model...')
num_classes = len(streamer.topics)
model = Sequential()
model.add(Dense(512, input_shape=(MAX_WORDS,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# Train model
with mlflow.start_run():
    mlflow.keras.autolog()

    history = model.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=1,
                        validation_split=0.1)
    score = model.evaluate(X_test, Y_test,
                           batch_size=BATCH_SIZE, verbose=1)

    model.save(KERAS_MODEL_FILE)

    artifacts = {
        'keras_model': KERAS_MODEL_FILE,
    }

    conda_env = 'conda.yaml'

    mlflow.pyfunc.log_model(
        'model', artifacts=artifacts, conda_env=conda_env,
        python_model=ModelWrapper(tokenizer, topics_encoder), code_path=['src']
    )

    # print some metrics
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    artifact_paths = save_samples(streamer.topics)
    for fp in artifact_paths:
        mlflow.log_artifact(fp, 'model')
