# Training Source Code

import mlflow
import logging
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s -%(levelname)s -%(message)s',
    handlers=[
        logging.FileHandler("mlops.log"),
        logging.StreamHandler()
    ]
)

logging.info('Starting model training process ......')

logging.info('Loading Data ......')
iris = load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2)

logging.info('Data loaded and split into training and test set ......')
with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)

    logging.info('Model Training completed ......')

    predictions = model.predict(X_test)
    accuracy = accuracy_score(Y_test,predictions)

    mlflow.log_metric("accuracy", accuracy)
    logging.info(f"Model Accuracy: {accuracy:.4f}")
