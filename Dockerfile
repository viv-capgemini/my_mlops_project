FROM python:3.9-slim
WORKDIR /app
# Install dependancies
COPY requirements.txt .
#Copy the rest of the code
COPY . .
# Command to run the model training script
CMD [ "python", "src/train.py" ]
