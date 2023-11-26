
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