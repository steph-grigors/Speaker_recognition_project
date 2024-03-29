#======================#
# Install, clean, test #
#======================#

install_requirements:
	@pip install -r requirements.txt

install:
	@pip install . -U

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr proj-*.dist-info
	@rm -fr proj.egg-info



#======================#
#          API         #
#======================#

run_api:
	uvicorn who_dis.api.fast:app --reload --port 8000


#======================#
#          GCP         #
#======================#

gcloud-set-project:
	gcloud config set project ${GCP_PROJECT}



#======================#
#         Docker       #
#======================#

# Local images - using local computer's architecture
# i.e. linux/amd64 for Windows / Linux / Apple with Intel chip
#      linux/arm64 for Apple with Apple Silicon (M1 / M2 chip)

docker_build_local:
	docker build --tag=$(GCR_IMAGE):local .

docker_run_local:
	docker run \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE_NAME):local

docker_run_local_interactively:
	docker run -it \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE_NAME):local \
		bash

# Cloud images - using architecture compatible with cloud, i.e. linux/amd64

docker_build:
	docker build \
		--platform linux/amd64 \
		-t $(GCR_REGION)/$(GCP_PROJECT)/$(GCR_IMAGE):prod .

# Alternative if previous doesn´t work. Needs additional setup.
# Probably don´t need this. Used to build arm on linux amd64
docker_build_alternative:
	docker buildx build --load \
		--platform linux/amd64 \
		-t $(GCR_REGION)/$(GCP_PROJECT)/$(GCR_IMAGE):prod .

docker_run:
	docker run \
		--platform linux/amd64 \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(GCR_REGION)/$(GCP_PROJECT)/$(GCR_IMAGE):prod

docker_run_interactively:
	docker run -it \
		--platform linux/amd64 \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(GCR_REGION)/$(GCP_PROJECT)/$(GCR_IMAGE):prod \
		bash

# Push and deploy to cloud

docker_push:
	docker push $(GCR_REGION)/$(GCP_PROJECT)/$(GCR_IMAGE):prod

docker_deploy:
	gcloud run deploy \
		--project $(GCP_PROJECT) \
		--image $(GCR_REGION)/$(GCP_PROJECT)/$(GCR_IMAGE):prod \
		--platform managed \
		--region europe-west1 \
		--env-vars-file .env.yaml
