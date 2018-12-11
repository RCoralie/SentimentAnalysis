MKFILE_DIR := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
MNT_DIR := $(abspath $(MKFILE_DIR)../)

USER := $(shell id -u)
DOCKER_CMD         = docker run

# Main docker image name
DOCKER_IMG         = sentimentanalysis-build:0.1

# Where to mount the base of the repository in the container
DOCKER_MOUNT_POINT = /sentimentAnalysis/

# Set up source code volume mount
DOCKER_VOLUME_MOUNT = -v $(MNT_DIR):$(DOCKER_MOUNT_POINT)

# Working directories
WORKING_DIRECTORY = $(DOCKER_MOUNT_POINT)
DOCKER_WD = /sentimentAnalysis

# Create optional environment variable and user flags to be passed to the docker command
ENV_VARS =
USER_FLAGS =
LOCAL_USER = --user=$(USER):$(USER)

# Set up final docker commands
DOCKER_BASE = $(DOCKER_CMD) --rm -it $(USER_FLAGS) $(ENV_VARS) $(DOCKER_VOLUME_MOUNT) -w $(WORKING_DIRECTORY)
DOCKER = $(DOCKER_BASE) $(DOCKER_IMG)

echo:
	@echo $(MNT_DIR)

image:
	docker build . -t $(DOCKER_IMG)

init:
	$(eval WORKING_DIRECTORY = $(DOCKER_WD))
	$(DOCKER) /sentimentAnalysis/docker/init.sh

train:
	$(eval WORKING_DIRECTORY = $(DOCKER_WD))
	$(DOCKER) /sentimentAnalysis/docker/trainOnTweets.sh

analyse:
	$(eval WORKING_DIRECTORY = $(DOCKER_WD))
	$(DOCKER) /sentimentAnalysis/docker/analyseOnTwitter.sh

test:
	$(eval WORKING_DIRECTORY = $(DOCKER_WD))
	$(DOCKER) /sentimentAnalysis/docker/runtests.sh

shell:
	$(DOCKER) bash
