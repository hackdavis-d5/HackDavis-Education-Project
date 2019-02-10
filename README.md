# HackDavis-Education-Project


## Back-end comunicating with Google Cloud


## Front-end
### Todo
1. Capture video stream
2. Measure difference between frames to determine if there has been enough change to send frame for OCR on google cloud
  * come up with some measure
3. Figure out the bounding boxes of the areas that contain text that needs OCR
  * clip out just the bounding boxes (don't want to send a giant image each time)
  * tag each counding box so we can place modified text back in the same box once we get OCR data
4. Forward images to back-end talking to google cloud
5. After getting OCR data back, display text with OpenDyslexic font in the correct spot in the image.
