#!/usr/bin/python

import socket
from _thread import *
import threading

host = ""
port = 8080
serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serv.bind((host, port))
serv.listen(5)

lock = threading.Lock()

def receive(conn):
    while True:
        data = conn.recv(1024)
        if not data:
            print("Terminated Connection")
            lock.release()
            break

        data = data.decode('ascii')
        conn.send(data[::-1].encode('ascii'))

    conn.close()

while True:
    conn, addr = serv.accept()

    lock.acquire()
    print("Received connection from " + addr[0])
    
    start_new_thread(receive, (conn,))
