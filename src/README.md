# Building and training a model for making predictions about house prices

### Setup virtual environment to train the model
- python3 -m venv venv && source venv/bin/activate
- pip install -r requirements.txt

### Train the model will result in generating model.pkl
- python src/model_training.py

### Track with DVC as its too large for git
- dvc add data/raw.csv model/model.pkl model/scaler.pkl
- git add *.dvc .dvc .dvcignore
- git commit -m "tracked artifacts"

### Push artifacts + Git metadata
- dvc push
- git push

### Build & Deploy Docker image
- docker build -t DOCKERHUB_USER/ml-model:v1 .
- docker push DOCKERHUB_USER/ml-model:v1

### Test
- docker run -p 80:80 DOCKERHUB_USER/ml-model:v1
- curl -X POST http://127.0.0.1:80/predict -d '{"features":[1,2,3]}'
