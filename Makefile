model:
	python -m sincnet.model

train:
	python -m training.train

token:
	python -m training.train_tokenizer

loader:
	python -m datasets.utils.waveform

compile:
	python -m datasets.compiler

dataset:
	python -m datasets.dataset

board:
	tensorboard --logdir=training/trainings/

qbench:
	python -m benchmark.quant_benchmark

.PHONY: stft
stft:
	python -m sincnet.stft
