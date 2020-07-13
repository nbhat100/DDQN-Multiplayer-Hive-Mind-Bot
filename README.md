# Pacman-Bot
Plays pacman on this website (Run this website while code is running and minimize the ide): http://www.free80sarcade.com/pacman.php  
**Architecture:**  
* **DDQN_Model.py** is a model that creates 2 from the *class ConvNet* in **Convolutional_Neural_Network.py**  
* **Get_environment** gets the environemnt by taking a screenshot into a numpy array.  
* **Gather_training_points** takes a screenshot of the reward and reads it using OCR.  
* **Main_Pacman_Bot.py** intergrates everything together  

***Note:*** *Pyautogui doesnt seem to interact with the js game but works with a flash game. Need to find better thing to train it on. Probably gonna directly train this on Diep.io* 
