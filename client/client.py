#!/usr/bin/python

import socket

host = "localhost"
port = 8080

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))

seed = "This is a test of "
sock.send(seed.encode('ascii'))
data = sock.recv(1024)
print(data)
sock.close()
