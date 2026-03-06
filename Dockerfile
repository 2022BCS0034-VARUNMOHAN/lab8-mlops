FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install pandas scikit-learn dvc[s3] numpy

CMD ["python", "src/train.py"]