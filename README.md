# DDQN Hive Mind Bot
Goal: plays multiplayer games with alts to efficiently clear enemies
Current status: working on making it play Diep.io
**Architecture:**  
* **DDQN_Model.py** is a model that creates 2 from the *class ConvNet* in **Convolutional_Neural_Network.py**  
* **Get_environment** gets the environemnt by taking a screenshot into a numpy array.  
* **Gather_training_points** takes a screenshot of the reward and reads it using OCR.  
* **Main_Pacman_Bot.py** intergrates everything together  

***Note:*** *Mouse and key intergration successful. Need to test model, and add Server side reward modeling* 
