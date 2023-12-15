python_version=$(python -c 'import platform; print(platform.python_version())')

if [[ $python_version == 3.8.10 ]]; then
    echo "Python 3.8.10 is installed"
else
    echo "Should have Python 3.8.10 installed but let's still try it"
fi

python -m venv env
source env/Scripts/activate
pip install swig
pip install cmake
cmake ffmpeg freeglut3-dev xvfb x11-utils
pip install pybullet
pip install tensor
pip install tensorboard
pip install gymnasium[box2d]
pip install stable-baselines3[extras]
pip install opencv-contrib-python

