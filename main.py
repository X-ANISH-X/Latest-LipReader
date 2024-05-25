from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory, jsonify
import os
from moviepy.editor import VideoFileClip

#
#
# import os
# import cv2
# import tensorflow as tf
# import numpy as np
# from typing import List
# from matplotlib import pyplot as plt
# import imageio
#
# tf.config.list_physical_devices('GPU')
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#     pass
#
# import gdown
#
# url = 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'
# output = 'data.zip'
# gdown.download(url, output, quiet=False)
# gdown.extractall('data.zip')

#



UPLOAD_FOLDER = 'static/videos'
lipReader = Flask(__name__)
lipReader.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
lipReader.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mkv', 'mov', 'mpg'}
lipReader.config['CONVERTED_FOLDER'] = 'converted'

if not os.path.exists(lipReader.config['UPLOAD_FOLDER']):
    os.makedirs(lipReader.config['UPLOAD_FOLDER'])

if not os.path.exists(lipReader.config['CONVERTED_FOLDER']):
    os.makedirs(lipReader.config['CONVERTED_FOLDER'])

@lipReader.route("/")
def home():
    return render_template("index.html")

@lipReader.route("/up")
def index():
    return render_template("index2.html")

@lipReader.route("/docs")
def docs():
    return render_template("docs.html")

@lipReader.route("/dnd")
def dnd():
    return render_template("dragndrop.html")

@lipReader.route('/upload1', methods=['POST'])
def upload_file1():
    if 'file' not in request.files:
        return jsonify(error='No file part'), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error='No selected file'), 400
    if file and file.filename.lower().endswith('.mpg'):
        original_filename = file.filename
        filepath = os.path.join(lipReader.config['UPLOAD_FOLDER'], original_filename)
        file.save(filepath)

        mp4_filename = f"{os.path.splitext(original_filename)[0]}.mp4"
        mp4_filepath = os.path.join(lipReader.config['CONVERTED_FOLDER'], mp4_filename)

        # Convert MPG to MP4
        try:
            clip = VideoFileClip(filepath)
            clip.write_videofile(mp4_filepath, codec='libx264')
        except Exception as e:
            return jsonify(error=str(e)), 500

        return jsonify(filename=mp4_filename)
    else:
        return jsonify(error='Invalid file format'), 400

@lipReader.route('/uploads1/<filename>')
def uploaded_file(filename):
    return send_from_directory(lipReader.config['CONVERTED_FOLDER'], filename)



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in lipReader.config['ALLOWED_EXTENSIONS']

@lipReader.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    # Inside the upload_file route
    if file and allowed_file(file.filename):
        filename = os.path.join(lipReader.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        return render_template('index2.html', video_filename=file.filename)


@lipReader.route('/uploads/<filename>')
def serve_video(filename):
    return send_from_directory(lipReader.config['UPLOAD_FOLDER'], filename)

#this doesn't work yet cause keras isn't working
@lipReader.route('/predict')
def predict_model():
    import os
    import pickle
    import cv2
    import tensorflow as tf
    import numpy as np
    from typing import List
    from matplotlib import pyplot as plt
    import imageio
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
    tf.config.list_physical_devices('GPU')
    import gdown
    #this section of code needs to be executed only once, can be deleted after first execution
    #--------------------------------------------------
    url = 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'
    output = 'data.zip'
    gdown.download(url, output, quiet=False)
    gdown.extractall('data.zip')
    #--------------------------------------------------

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
    def load_video(path: str) -> List[float]:
        cap = cv2.VideoCapture(path)
        frames = []
        for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, frame = cap.read()
            frame = tf.image.rgb_to_grayscale(frame)
            frames.append(frame[190:236, 80:220, :])
        cap.release()

        mean = tf.math.reduce_mean(frames)
        std = tf.math.reduce_std(tf.cast(frames, tf.float32))
        return tf.cast((frames - mean), tf.float32) / std

    vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

    char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
    num_to_char = tf.keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
    )

    def load_alignments(path: str) -> List[str]:
        with open(path, 'r') as f:
            lines = f.readlines()
        tokens = []
        for line in lines:
            line = line.split()
            if line[2] != 'sil':
                tokens = [*tokens, ' ', line[2]]
        return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

    def load_data(path: str):
        path = bytes.decode(path.numpy())
        # file_name = path.split('/')[-1].split('.')[0]
        # File name splitting for windows
        file_name = path.split('\\')[-1].split('.')[0]
        video_path = os.path.join('data', 's1', f'{file_name}.mpg')
        alignment_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')
        frames = load_video(video_path)
        alignments = load_alignments(alignment_path)

        return frames, alignments

    test_path = '.\\data\\s1\\bbal6n.mpg'

    tf.convert_to_tensor(test_path).numpy().decode('utf-8').split('\\')[-1].split('.')[0]

    frames, alignments = load_data(tf.convert_to_tensor(test_path))

    #plt.imshow(frames[40])

    tf.strings.reduce_join([bytes.decode(x) for x in num_to_char(alignments.numpy()).numpy()])

    def mappable_function(path: str) -> List[str]:
        result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
        return result

    from matplotlib import pyplot as plt

    data = tf.data.Dataset.list_files('./data/s1/*.mpg')
    data = data.shuffle(500, reshuffle_each_iteration=False)
    data = data.map(mappable_function)
    data = data.padded_batch(2, padded_shapes=([75, None, None, None], [40]))
    data = data.prefetch(tf.data.AUTOTUNE)
    # Added for split
    train = data.take(450)
    test = data.skip(450)

    frames, alignments = data.as_numpy_iterator().next()

    sample = data.as_numpy_iterator()
    val = sample.next()
    imageio.mimsave('./animation.gif', val[0][0], fps=10)
    tf.strings.reduce_join([num_to_char(word) for word in val[1][0]])


    model = Sequential()
    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'))

    yhat = model.predict(val[0])

    tf.strings.reduce_join([num_to_char(x) for x in tf.argmax(yhat[0], axis=1)])

    tf.strings.reduce_join([num_to_char(tf.argmax(x)) for x in yhat[0]])

    sample = load_data(tf.convert_to_tensor('.\\data\\s1\\bbaf3s.mpg'))

    def scheduler(epoch, lr):
        if epoch < 30:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    def CTCLoss(y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss


    def on_epoch_end(self, epoch, logs=None) -> None:
        data = self.dataset.next()
        yhat = self.model.predict(data[0])
        decoded = tf.keras.backend.ctc_decode(yhat, [75, 75], greedy=False)[0][0].numpy()
        for x in range(len(yhat)):
            print('Original:', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8'))
            print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
            print('~' * 100)

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001), loss=CTCLoss)

    checkpoint_callback = ModelCheckpoint(os.path.join('models', 'checkpoint'), monitor='loss', save_weights_only=True)

    schedule_callback = LearningRateScheduler(scheduler)

    example_callback = ProduceExample(test)

    #this section needs to be executed only once
    #------------------------------------------------------------------
    url = 'https://drive.google.com/uc?id=1vWscXs4Vt0a_1IH1-ct2TCgXAZT-N3_Y'
    output = 'checkpoints.zip'
    gdown.download(url, output, quiet=False)
    gdown.extractall('checkpoints.zip', 'models')

    model.load_weights('models/checkpoint')
    #---------------------------------------------------------------------

    sample = load_data(tf.convert_to_tensor('.\\data\\s1\\bbaf3s.mpg'))

    yhat = model.predict(tf.expand_dims(sample[0], axis=0))

    decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()

    print('~' * 10, 'PREDICTIONS')
    ans = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded]
    print(ans)
    #Ans is preferred to be outputted in the text box of drag and drop page in the website, this funtion is just in order to check if ans will be outputted in /predict route, if that works we will proceed to add it in textbox.
    return ans

if __name__ == "__main__":
    lipReader.run(debug=True)
