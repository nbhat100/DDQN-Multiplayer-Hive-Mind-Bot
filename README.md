# Pacman-Bot
Plays pacman
**Architecture:**  
* **DDQN_Model.py** is a model that creates 2 from the *class ConvNet* in **Convolutional_Neural_Network.py**  
* **Get_environment** gets the environemnt by taking a screenshot into a numpy array.  
* **Gather_training_points** takes a screenshot of the reward and reads it using OCR.  
* **Main_Pacman_Bot.py** intergrates everything together  

***Note:*** *In DDQN_Model, there is some code that causes the entire folder to be deleted in lines 40-50* 
