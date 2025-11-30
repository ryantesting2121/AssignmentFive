API_IMAGE=sentiment-api
MONITOR_IMAGE=sentiment-monitor
NETWORK=mlops-net
VOLUME=mlops-logs

build:
	docker build -t $(API_IMAGE) ./api
	docker build -t $(MONITOR_IMAGE) ./monitoring

run:
	docker network create $(NETWORK) || true
	docker volume create $(VOLUME) || true
	docker run -d --name api --network $(NETWORK) -v $(VOLUME):/app/logs -p 8000:8000 $(API_IMAGE)
	docker run -d --name monitoring --network $(NETWORK) -v $(VOLUME):/app/logs -p 8501:8501 $(MONITOR_IMAGE)

clean:
	docker rm -f api monitoring || true
	docker rmi $(API_IMAGE) $(MONITOR_IMAGE) || true
	docker network rm $(NETWORK) || true
	docker volume rm $(VOLUME) || true
