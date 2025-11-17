FROM python:3.10

WORKDIR /app

COPY . /app
     
RUN ls -l /app

RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --force-reinstall intersystems-irispython==5.2.1 


# ENV PYTHONPATH="usr/irissys/lib/python3.10/site-packages:/opt/conda/lib/python3.10/site-packages:/app"

EXPOSE 8000


# CMD ["sh", "-c", "python load_data.py && python app.py --debug true"]
# CMD ["sh", "-c", "python app.py --debug true"]


