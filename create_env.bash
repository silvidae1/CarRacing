python_version=$(python -c 'import platform; print(platform.python_version())')

if [[ $python_version == 3* ]]; then
    echo "Python 3 is installed"
elif [[ $python_version == 2* ]]; then
    echo "Python 2 is installed"
else
    echo "Python is not installed or could not determine the version"
fi

python -m venv env
source env/Scripts/activate
pip install swig
pip install cmake
cmake ffmpeg freeglut3-dev xvfb x11-utils
pip install pybullet
pip install gymnasium[box2d]
pip install stable-baselines3[extras]
pip install tensor
pip install tensorboard
