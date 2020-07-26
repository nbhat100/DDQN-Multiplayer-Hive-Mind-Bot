#!/usr/bin/python
#65432

import socket
import numpy as np
from io import BytesIO
from struct import *
import zlib
from get_environment import GetEnv
from gather_training_points import readReward
import pydirectinput
import threading
import random


Frames = 5
InFrameDim = (128, 1031, 0, 1920)
InputShape = (270, 480, Frames)
TotalStepLimit = 5000000
ActionSpace = ("w", "a", "s", "d", "enter", "e")
MouseActionSpace = [[1061, 581], [1053, 619], [1031, 651], [999, 673], [961, 681], [922, 673], [890, 651], [868, 619], [861, 581], [868, 542], [890, 510], [922, 488], [961, 481], [999, 488], [1031, 510], [1053, 542]]
ScoreInDim = (1014, 1029, 875, 1045)
ScoreOutDim = (170, 16)



class Environment:
    def __init__(self):
        self.env = GetEnv(inDim=InFrameDim, outDim=(InputShape[1], InputShape[0]))
        self.out = []
        for i in range(5):
            self.out.append(self.env.takeImage(4, "None"))
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True
        thread.start()

    def run(self):
        while True:
            self.out.append(self.env.takeImage(4, "None"))
            self.out.pop(0)

    def getEnvironment(self):
        return self.out

    def step(self, action, prev_action):
        print(f"action: {action}")
        print(f"Mouse Action: {MouseActionSpace[action[1]]}")
        upgrade = random.randrange(1, 9)
        pydirectinput.press(str(upgrade))
        pydirectinput.keyUp(ActionSpace[prev_action[0]])
        pydirectinput.keyDown(ActionSpace[action[0]])
        pydirectinput.moveTo(MouseActionSpace[action[1]][0], MouseActionSpace[action[1]][1])

class Client:
    def __init__(self, host="", port=15187, ident=0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.id = ident

    def send_image_array(self, cur_state):
        data = zlib.compress(np.array(cur_state).tobytes())
        print(cur_state.dtype)
        print(len(data))
        self.sock.sendall(pack('>Q', len(data)))
        self.sock.sendall(data)
        self.sock.recv(1)

    def send_int(self, num):
        self.sock.sendall(pack('>Q', num))

    def main(self):
        run = 0
        total_step = 0
        prev_score = 0
        prev_action = [0, 0]
        while True:
            run += 1
            env = Environment()
            current_state = env.getEnvironment()
            step = 0
            print(total_step)
            score = readReward()
            while total_step <= TotalStepLimit:
                total_step += 1
                step += 1
                print(f"Step: {total_step}")
                print(np.array(current_state).shape)
                print(np.array(current_state).size)
                print(np.array(current_state).ravel())
                self.send_image_array(np.array(current_state))

                length = unpack('>Q', self.sock.recv(8))[0]
                data = b''
                while len(data) < length:
                    to_read = length - len(data)
                    data += self.sock.recv(min(to_read, 4096))
                action = np.frombuffer(data, dtype="int64")
                action = action[:6]
                print(action)
                action = action.reshape(3, 2)
                action = action[self.id]
                env.step(action, prev_action)
                prev_action = action
                current_score = score.mainLoop((965, 982, 895, 1025), (130, 17))
                reward = current_score - prev_score
                self.send_string(reward)


client = Client(host="2.tcp.ngrok.io", port=12749, ident=0)
client.main()
