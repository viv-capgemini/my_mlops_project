FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first to take advantage of Docker layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && \
	pip install -r requirements.txt
# Do not run model training during image build. If you need to train,
# run `python model_training.py` locally or in a dedicated training pipeline.
# Copy application code
COPY . .

EXPOSE 5001

# Run the app from the working directory
CMD ["python", "app.py"]

