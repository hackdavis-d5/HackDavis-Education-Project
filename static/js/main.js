document.addEventListener("DOMContentLoaded", function(event) { 
    const video = {
        init: function() {
            this.cacheDDM();
            this.initVideo();
            this.initCanvas();
            this.initContext();
        },

        cacheDDM: function() {
            this.container = document.querySelector('#container');
            this.videoElement = document.querySelector("video");
            this.canvas = document.querySelector("canvas");
            
        },

        initVideo: function() {

            const onSuccess = (stream) => {
                this.videoElement.srcObject = stream;
            };

            const onError = () => {
                console.log("Something went wrong!");
            };

            if (navigator.mediaDevices.getUserMedia) {       
                navigator.mediaDevices.getUserMedia({video: true})
                .then(onSuccess)
                .catch(onError);
            }
        },
        
        initCanvas: function() {
            this.canvas.width = 320;
            this.canvas.height = 320;
        },

        initContext: function() {
            this.context = this.canvas.getContext('2d');
            this.context.drawImage(this.videoElement, 0, 0, this.canvas.width, this.canvas.height);
        },

        getFrameAsBase64URL: function() {
            this.context.drawImage(this.videoElement, 0, 0, this.canvas.width, this.canvas.height);
            return this.canvas.toDataURL('image/jpeg');
        }
    };

    const socket = {
        init: function() {
            this.initSocket();
        },

        initSocket: function() {
            this.sock = io.connect('http://localhost:5000');
            this.sock.on('ocrComplete', (words) => {
                if (words) {
                    const text = words.reduce((acc, curr) => `${acc} ${curr}`, '');
                    document.querySelector('#output').innerHTML = text;
                }
            }); 
        },

        send: function(event, message) {
            this.sock.emit(event, message);
        }
    };

    video.init();
    socket.init();

    setInterval(function() {
        socket.send('rawFrame', video.getFrameAsBase64URL());
    }, 1000);
});    