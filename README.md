# pistolDetectionYolov5
This app is developed with Django4.0.5 and yolov5.

requirements
python 3.10.4

You need a  virtual environment with python:

for windows: 
run powershell or cmd
- python -m venv venv
- ./venv/scripts/activate.ps1
if you have a nvidia gpu - pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

if you don't have nvidia gpu
- pip install -r requirements.txt
- python manage.py runserver
 

mac or linux:
- python3 -m venv venv
- source venv/scripts/activate
if you have a nvidia gpu -pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

if you don't have nvidia gpu
- pip install -r requirements.txt
- python3 manage.py runserver
