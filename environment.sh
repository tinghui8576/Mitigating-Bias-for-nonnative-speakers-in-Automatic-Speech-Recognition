#Load preinstalled modules
module load python3/3.10.12
module load numpy/1.24.3-python-3.10.12-openblas-0.3.23
module load cudnn/v8.9.1.23-prod-cuda-11.X 
module load pandas/2.0.2-python-3.10.12
module load cuda/11.8

#Create a virtual environment for Python3
python3 -m venv /work3/s232855/whisper

#Activate virtual environment
source /work3/s232855/whisper/bin/activate

#If pip3 fails, use: which pip3, to make sure it is the one in the virutal environment.
#which pip3
pip3 install -r requirements.txt
