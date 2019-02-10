  
  var video=document.querySelector('video');
  var canvas=document.querySelector('canvas');
  var context=canvas.getContext('2d');
  var w,h,ratio;
  //add loadedmetadata which will helps to identify video attributes

  video.addEventListener('loadedmetadata', function() {
    ratio = video.videoWidth/video.videoHeight;
    w = video.videoWidth-100;
    h = parseInt(w/ratio,10);
    canvas.width = w;
    canvas.height = h;
  },false);

  function snap() {
    context.fillRect(0,0,w,h);
    context.drawImage(video,0,0,w,h);
  }
  setInterval(snap, 1000);

Filters = {};
Filters.getPixels = function() {
  return context.getImageData(0,0,canvas.width,canvas.height);
};

Filters.filterImage = function(filter, image, var_args) {
    var args = [this.getPixels(image)];
    for (var i=2; i<arguments.length; i++) {
      args.push(arguments[i]);
    }
    return filter.apply(null, args);
  };

Filters.threshold = function(pixels, threshold) {
    var d = pixels.data;
    for (var i=0; i<d.length; i+=4) {
      var r = d[i];
      var g = d[i+1];
      var b = d[i+2];
      var v = (0.2126*r + 0.7152*g + 0.0722*b >= threshold) ? 255 : 0;
      d[i] = d[i+1] = d[i+2] = v
    }
    return pixels;
  };