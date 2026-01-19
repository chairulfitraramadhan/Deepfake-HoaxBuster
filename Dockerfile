FROM python:3.9

WORKDIR /code

RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

RUN mkdir -p uploads && chmod 777 uploads

CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]