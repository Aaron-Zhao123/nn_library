DATA_DIR=/local/scratch/ssd
TRAIN_DIR=/local/scratch/yaz21/tmp2
CPU_DATA_DIR=/Users/aaron/Projects/NN_models/datasets/flowers
GPU_DATA_DIR=/local/scratch/ssd
TRAIN_DIR=./tmp

run: gpu_training.py
	CUDA_VISIBLE_DEVICES=0,1 python gpu_training.py --num_preprocess_threads=4 --num_gpus=2 --train_dir=$(TRAIN_DIR) --data_dir=$(GPU_DATA_DIR) --batch_size=64

clean:
	rm -rf *.pyc
	find . -name "*.pyc" -delete

cpurun: cpu_train.py
	python cpu_train.py --train_dir=$(TRAIN_DIR) --data_dir=$(CPU_DATA_DIR)

git-add:
	git add -A
	git commit -m"mobilenet first commit"
	git push

git-fetch:
	git fetch
	git merge
