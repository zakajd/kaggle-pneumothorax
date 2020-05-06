.PHONY: all clean load preprocess train inference

PYTHON = python3

# all: results/solution/solution.csv

preprocess: load data/interim/train_val.csv 

## Load datasets to data/raw folder
data/raw :load
	kaggle datasets download seesee/siim-train-test -p data/raw
	unzip \
		-q data/raw/siim-train-test.zip\
		-d data/raw


data/interim/masks data/interim/images data/interim/train_val.csv: src/data/preprocess.py
	$(PYTHON) $< \
		--root data/raw \
		--output_path data/interim \
		--create_masks \
		--train_val_split \
		--num_folds 5 \
		# --use_clahe \
		# -- dilate_mask\

## Delete everything except raw data and code
# clean:
# 	find . -type f -name "*.py[co]" -delete
# 	find . -type d -name "__pycache__" -delete
# 	rm -r data/processed
# 	rm -r logs/
# 	rm -r results/


