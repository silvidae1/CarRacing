### Run the environment

If it is the first time running the code, then make sure to first setup the environment as explained below!

To run the environment, first you need to source the environment:
```Shell
source env/Scripts/activate
```

Then just run either Training.py or run.py with:
```Shell
python run.py
```

In run.py make sure to specify the trained model you want by changing the PPO.load path.
In Training.py, change your run name to one that wasn't used before, otherwise it might erase previous logs!

Have fun!

### Setup environment

Before doing the following commands, make sure to:
 - Have Microsoft C++ Build Tools installed
 - Have python 3.8.10 version installed

Assuming you are running on windows, these commands should be used on *git bash*. 
```Shell
python -m venv env
source env/Scripts/activate
pip install swig
pip install cmake
pip install pybullet
pip install gymnasium[box2d]
pip install stable-baselines3[extras]
pip install tensor
pip install tensorboard
```


### Todo's
 - Add FrameStack
 - Add GPU
 - Add entropy value