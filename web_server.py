import cv2
import numpy as np
from flask import Flask, Response
import socket
from threading import Thread, Event


def getServerHost():
    testSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    testHost = '10.0.0.0'
    testPort = 0
    testSock.connect((testHost, testPort))
    host = testSock.getsockname()[0]
    testSock.close()
    return host

def concat_images(imgs, direction='h', bg_color=(0, 0, 0)):
    """Склеивает изображения с добавлением фона при разных размерах"""
    if direction == 'h':
        max_h = max(img.shape[0] for img in imgs)
        total_w = sum(img.shape[1] for img in imgs)
        result = np.full((max_h, total_w, 3), bg_color, dtype=np.uint8)
        x = 0
        for img in imgs:
            result[0:img.shape[0], x:x+img.shape[1]] = img
            x += img.shape[1]
    else:  # vertical
        max_w = max(img.shape[1] for img in imgs)
        total_h = sum(img.shape[0] for img in imgs)
        result = np.full((total_h, max_w, 3), bg_color, dtype=np.uint8)
        y = 0
        for img in imgs:
            result[y:y+img.shape[0], 0:img.shape[1]] = img
            y += img.shape[0]
    return result

class WebServer:
    def __init__(self, framesReceiversDict, streamingPort):
        self.app = Flask(__name__)
        self.framesReceiversDict = framesReceiversDict
        self.framesBufferThreads = {}
        self.streamsIsRunning = True

        self.InitRoutes()

        self.streamingPort = streamingPort
        self.host = getServerHost()

        self.eventsDict = {}
        self.lastFramesDict = {}
        for ip in self.framesReceiversDict:
            self.eventsDict[ip] = Event()
            self.lastFramesDict[ip] = None

    def Start(self):
        serverThread = Thread(
            target=self.app.run, args=(self.host, self.streamingPort),
            kwargs={'debug': False, 'threaded': True}, daemon=True)
        for ip in self.framesReceiversDict:
            self.framesBufferThreads[ip] = Thread(target=self.FrameReader,
                                                  args=[ip])
            self.framesBufferThreads[ip].start()

        serverThread.start()

    def FrameReader(self, ip):
        receiver = self.framesReceiversDict[ip]
        while self.streamsIsRunning:
            if not receiver.poll(0.1):
                continue
            self.lastFramesDict[ip] = receiver.recv()
            self.eventsDict[ip].set()

    def GenerateFrames(self, ip):
        event = self.eventsDict[ip]
        while True:
            event.wait(0.1)
            if not event.is_set():
                continue
            event.clear()
            frame, map = self.lastFramesDict[ip]
            frame = concat_images([frame, map])
            ret, buffer = cv2.imencode('.jpg', frame)
            frameBytes = buffer.tobytes()
            if frameBytes is None:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frameBytes + b'\r\n')

    def InitRoutes(self):
        @self.app.route('/')
        def index():
            return "RTSP Server"

        for ip in self.framesReceiversDict:
            @self.app.route(f'/video_feed/{ip}', endpoint=ip)
            def videoFeed(ip=ip):
                return Response(self.GenerateFrames(ip),
                                mimetype=
                                'multipart/x-mixed-replace; boundary=frame')

    def Close(self):
        self.streamsIsRunning = False
        for ip in self.framesReceiversDict:
            self.framesBufferThreads[ip].join()