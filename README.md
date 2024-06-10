# milvus-bge-m3

Repo for starting milvus standalone or as a cluster

Requirements

- Docker
- Docker Compose
- Python 3

**IMPORTANT**
If you have started one of the docker compose files and you are planning to start the other, delete the volumes folder otherwise it will not work.


## Starting Milvus Cluster

docker compose command to start standalone **docker-compose -f docker-compose-v2.4-cluster.yml up -d** wait for the containers to be healthy with **docker ps**

## Starting Milvus Standalone

docker compose command to start standalone **docker-compose -f docker-compose-v2.4-standalone.yml up -d** wait for the containers to be healthy with **docker ps**

## Running test script to check if Milvus is working

- Install requirements with **pip3 install -r requirements.txt**
- Run the script with **python3 hello_milvus.py**