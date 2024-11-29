FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y python3 python3-pip libopencv-dev
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

# Copy application files
COPY . /app
WORKDIR /app

CMD ["python3", "app.py"]
