FROM ubuntu:latest
WORKDIR /app
copy . /app

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get install -y tzdata
RUN apt-get install -y libglib2.0-dev
RUN apt-get install -y libgtk2.0-0
RUN apt-get install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev

RUN pip3 install -r requirements.txt

#EXPOSE 8501
ENV PORT 8501
#CMD ["streamlit", "run", "app.py"]
CMD streamlit run --server.port $PORT app.py
