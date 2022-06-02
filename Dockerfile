FROM python:3.9.6
WORKDIR /RBS-MLOps-Challenge
ADD . /RBS-MLOps-Challenge
RUN pip install --upgrade pip
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install -r requirements.txt
CMD python app.py
