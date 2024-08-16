FROM python:3.9.13-slim

WORKDIR /app

COPY . /app

RUN pip install gunicorn
RUN pip install "numpy<2"
RUN pip install torch==1.10.0+cpu torchvision==0.11.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV FLASK_APP=LeafLab.py
ENV FLASK_RUN_HOST=0.0.0.0

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "LeafLab:app"]
