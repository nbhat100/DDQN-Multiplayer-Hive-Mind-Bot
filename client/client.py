#!/usr/bin/python

import socket
import numpy as np
from io import BytesIO
from struct import *
import zlib

class Client:
    def __init__(self, host="", port=65432):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    def send_image_array(self, img):
        data = zlib.compress(img.tobytes())
        self.sock.sendall(pack('>Q', len(data)))
        self.sock.sendall(data)
        self.sock.recv(1)
        return unpack('>Q', self.sock.recv(8))

client = Client()
client.send_image_array(np.array([[[.1] * 3] * 1920] * 1080))
