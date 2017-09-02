import numpy as np
import cv2
import time
import datetime
import SocketServer, SimpleHTTPServer, thread, ssl

# chrome://flags/#allow-insecure-localhost


count = 0
class NextVideoHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/script.js"):
            script = "next(" + str(count) + ")";
            self.send_response(200)
            self.send_header("Content-type", "application/javascript")
            self.send_header("Content-length", len(script))
            self.end_headers()
            self.wfile.write(script)
        else:
            script = """
<html><body><pre>
wasCount = 0;
next = function(count) {
  if (count &gt; wasCount) {
    wasCount = count
    document.getElementsByClassName("ytp-next-button")[0].click() 
  }
}

var head = document.getElementsByTagName('head')[0]
var script = null;
var cnt = 0;
var iv = setInterval(function() {
  if (script) {
    head.removeChild(script);
  }
  script = document.createElement('script');
  script.src = "https://localhost:8334/script.js";
  head.appendChild(script);
}, 2000);

"""
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.send_header("Content-length", len(script))
            self.end_headers()
            self.wfile.write(script)


class ReusableTCPServer(SocketServer.TCPServer):
    allow_reuse_address = True

httpd = ReusableTCPServer(("0.0.0.0", 8334), NextVideoHandler )
httpd.socket = ssl.wrap_socket (httpd.socket, certfile='server.pem', keyfile='server.key', server_side=True)
thread.start_new_thread(httpd.serve_forever, ())


cap = cv2.VideoCapture(1)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
today = datetime.datetime.now()

# define range of purple color in HSV
lower_purple = np.array([90,50,50])
upper_purple = np.array([140,255,255])

# Create empty points array
points = []

# Get default camera window size
ret, frame = cap.read()
Height, Width = frame.shape[:2]
frame_count = 0
kernel = np.ones((3,3), np.uint8)
thresh = Height / 4

out1 = cv2.VideoWriter('/home/oleksiyp/Videos/' + today.strftime('%Y-%m-%d:%H:%M:%S') + '-raw.avi', fourcc, 25.0, (Width, Height))
out2 = cv2.VideoWriter('/home/oleksiyp/Videos/' + today.strftime('%Y-%m-%d:%H:%M:%S') + '-rec.avi', fourcc, 25.0, (Width, Height))

wasMatch = False

while cap.isOpened():
    #lower_purple = np.array([ang,0,0])
    #upper_purple = np.array([ang+20,255,255])
    #ang = ang + 1
    # Capture webcame frame
    ret, frame = cap.read()
    if not ret:
        break


    out1.write(frame)
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv_img, lower_purple, upper_purple)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
   
    mask = cv2.dilate(mask, kernel, iterations = 1)
    #mask = cv2.erode(mask, kernel, iterations = 4)
    
    #frame = cv2.bitwise_and(frame, frame, mask=mask);
 
    # Find contours, OpenCV 3.X users use this line instead
    _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create empty centre array to store centroid center of mass
    center = int(Height/2), int(Width/2)
 
    radius = 0

    match = False
    if len(contours) > 0:
        
        # Get the largest contour and its center 
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        try:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        except:
            center =   int(Height/2), int(Width/2)
           
        # Allow only countors that have a larger than 15 pixel radius
        match = radius > 40
        if match:
            
            # Draw cirlce and leave the last center creating a trail
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 0, 255), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

    # Log center points
    points.append(center)

    # loop over the set of tracked points
    if match:
        for i in range(1, len(points)):
            try:
                cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)
            except:
                pass

        # Make frame count zero
        frame_count = 0
    else:
        # Count frames
        frame_count += 1

        # If we count 10 frames without object lets delete our trail
        if frame_count == 10:
            points = []
            # when frame_count reaches 20 let's clear our trail
            frame_count = 0

    cv2.line(frame, (0, thresh), (Width, thresh), (255, 255, 0), 2)

    match = match and center[1] < thresh

    if match and wasMatch != match:
        count = count + 1

    wasMatch = match

    # Display our object tracker
    frame = cv2.flip(frame, 1)
    output_text = "Count: " + str(count)

    cv2.putText(frame, output_text, (50,100),
                cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)

    out2.write(frame)
    cv2.imshow("Object Tracker", frame)

    if cv2.waitKey(1) == 10: #13 is the Enter Key
        break

# Release camera and close any open windows
cap.release()
out1.release()
out2.release()
cv2.destroyAllWindows()
httpd.socket.close()

