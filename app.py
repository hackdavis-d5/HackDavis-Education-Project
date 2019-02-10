from flask import Flask
from flask import render_template
from flask_socketio import SocketIO, emit
from util.text_detection import data_uri_to_cv2_img, segmentImage

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def socketIOConnect():
    emit('my response', {'data': 'Connected'})

@socketio.on('disconnect')
def socketIODisconnect():
    print('Client disconnected')

@socketio.on('rawFrame')
def handleImageFromClient(message):
    segmentImage(message)


if __name__ == '__main__':
    socketio.run(app)