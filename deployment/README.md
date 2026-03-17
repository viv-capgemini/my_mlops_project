# Deploying model in k8s

### Creating deployment
- Make sure you have Kubernetes installed and creating a namespace to run the aplication will be a good idea.
- My deployment creates 3 replicas of the model and a service with an end pint to used to predict house prices.
- I am using the model from https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
### Using model to predict a house price.
- Find end point od the service and use it to send a POST request for house price using you own features.
- `curl -X POST http://127.0.0.1:5001/predict -H "Content-Type: application/json" -d '{"features":[8.3252, 41, 6.984, 1.023, 322, 2.565, 37.88, -122.23]}'`
