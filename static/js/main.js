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
            this.canvas.width = 1080;
            this.canvas.height = 720;
        },

        initContext: function() {
            this.context = this.canvas.getContext('2d');
            this.context.drawImage(this.videoElement, 0, 0, this.canvas.width, this.canvas.height);
        },

        getFrameAsBase64URL: function() {
            this.context.drawImage(this.videoElement, 0, 0, this.canvas.width, this.canvas.height);
            return this.canvas.toDataURL('image/jpeg');
        },

        drawOverlayWithText: function(text, vertices) {
            const width = vertices[1].x - vertices[0].x;
            const height = vertices[2].y - vertices[0].y;
            this.context.clearRect(
                vertices[0].x, 
                vertices[0].y, 
                width,
                height
            );

            this.context.fillText(text, vertices[3].x, vertices[3].y, width - 10);
            this.context.font = `${height}px OpenDyslexic`;
            this.context.stroke();
        }
    };

    const socket = {
        init: function() {
            this.initSocket();
        },

        initSocket: function() {
            this.sock = io.connect('http://localhost:5000');
            this.sock.on('ocrComplete', ({words, vertices}) => {
                if (words) {
                    for (let i = 0; i < words.length; ++i) {
                        const word = words[i];
                        const box = vertices[i];
                        video.drawOverlayWithText(word, box);
                    }
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
    }, 200);
});    