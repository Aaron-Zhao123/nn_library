DATA_DIR=/local/scratch/ssd
TRAIN_DIR=/local/scratch/yaz21/tmp
DATA_DIR=/Users/aaron/Projects/NN_models/datasets/flowers
TRAIN_DIR=./tmp

run: gpu_train.py
	CUDA_VISIBLE_DEVICES=2,3 python gpu_train.py --num_preprocess_threads=4 --num_gpus=2 --train_dir=$(TRAIN_DIR) --data_dir=$(DATA_DIR)

clean:
	rm -rf *.pyc
	find . -name "*.pyc" -delete

cpurun: cpu_train.py
	python cpu_train.py --train_dir=$(TRAIN_DIR) --data_dir=$(DATA_DIR)

git-add:
	git add -A
	git commit -m"mobilenet first commit"
	git push

git-fetch:
	git fetch
	git merge
