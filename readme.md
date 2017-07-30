##### Inspect GPU utilization

# NN Lib 
A general neural network library to run network training on multi-gpu machines.
## Dependencies
1. Tensorflow

## Run

**gpu**: `make run`

**cpu**: `make cpurun`

**hyper-parameters**:

1. CUDA_VISIBLE_DEVICES, defines how many GPUs to use, defining this avoids affecting memories
2. num_threads, preprocessing threads
3. num_gpus, gpu threads
4. train_dir
5. data_dir
6. batch_size, number of elements per batch

**Inspect gpu utilization**: ```watch -n 0.5 nvidia-smi```


## License
MIT License

Copyright (c) 2017 Aaron-Zhao123

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
