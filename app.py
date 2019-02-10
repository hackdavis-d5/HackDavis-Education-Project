from flask import Flask
from flask import render_template
from flask_socketio import SocketIO, emit
from util.text_detection import segmentImage

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
    words = segmentImage(message)
    if words != []:
        emit('ocrComplete', words)


if __name__ == '__main__':
    socketio.run(app)