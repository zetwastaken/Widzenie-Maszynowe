PYTHON ?= python3.11
PIP ?= $(PYTHON) -m pip
INPUT ?= video.mp4
OUTPUT ?= out
STEP ?= 1

.PHONY: install run

install:
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) main.py --input $(INPUT) --output $(OUTPUT) --step $(STEP)

