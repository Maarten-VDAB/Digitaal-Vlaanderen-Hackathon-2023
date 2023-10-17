FROM python:3.10
ENV PYTHONUNBUFFERED=1
COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
COPY . /opt/app
EXPOSE 8000
CMD ["python","main.py"]