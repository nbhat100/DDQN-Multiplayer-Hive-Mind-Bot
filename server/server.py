#!/usr/bin/python

from model import *
import socket
from threading import *
import threading
from struct import *
from io import BytesIO
import zlib
import sys
import numpy as np

class Server:
    def __init__(self, host="", port=65432, input_shape=(1920, 1080, 3), learning_rate=0.01):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.buff = np.array([])
        self.input_shape = input_shape
        self.model = createModel(input_shape)
        self.model = CustomModel(self.model.inputs, self.model.outputs)
        adam = Adam(lr=learning_rate)
        self.model.compile(optimizer=adam)

    def handle_image(self, conn):
        try:
            while True:
                length = unpack('>Q', conn.recv(8))[0]
                data = b''
                while len(data) < length:
                    to_read = length - len(data)
                    data += conn.recv(min(to_read, 4096))

                data = zlib.decompress(data)
                data = np.frombuffer(data)
                data = data.reshape(self.input_shape)
                self.buff = np.append(self.buff, data).reshape(np.append(-1, self.input_shape))
                print(self.buff.shape)

                assert len(b'\00') == 1
                conn.send(b'\00')
        except BrokenPipeError as e:
            self.sock.sendall(b'\01')
            print("Server shut down by user")
            sys.exit()
        except Exception as e:
            print(e)
            self.sock.close()
        finally:
            conn.close()

    def start(self):
        try:
            self.sock.bind((self.host, self.port))
            self.sock.listen(5)
            while True:
                conn, addr = self.sock.accept()

                thread = Thread(target=self.handle_image, args=(conn,))
                thread.start()
                print(threading.active_count())
                if self.buff.shape[0] >= 3:
                    self.sock.sendall(self.model.predict(self.buff))
        except KeyboardInterrupt as e:
            self.sock.sendall(b'\01')
            print("Server shut down by user")
            self.sock.close()
            sys.exit()
        finally:
            self.sock.close()

if __name__ == '__main__':
    server = Server()
    server.start()
