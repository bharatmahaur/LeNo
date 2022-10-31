## LeNo-ResNet: ResNet with Norm-Preserving Downsampling and Identity Blocks

Pytorch implementation of LeNo-ResNet proposed in:<br />
Bharat Mahaur et al. "Improved Residual Network Based on Norm-Preservation for Visual Recognition." Neural Networks 2022.

Please find the paper here: [https://doi.org/10.1016/j.neunet.2022.10.023](https://doi.org/10.1016/j.neunet.2022.10.023).


### Requirements

Install PyTorch and ImageNet dataset following the official [PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

A fast alternative (without installing PyTorch and other deep learning libraries) is to use [NVIDIA-Docker](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/pullcontainer.html#pullcontainer).


### Training and Validation
To train a model (for instance, LeNo-ResNet with 50 layers) using DataParallel run `main.py`; you need to provide `result_path` (the directory path where to save the results and logs) and the `--data` (the path to the ImageNet dataset): 

```bash
result_path=/your/path/to/save/results/and/logs/
mkdir -p ${result_path}
python main.py \
--data /your/path/to/ImageNet/dataset/ \
--result_path ${result_path} \
--arch lenoresnet \
--model_depth 50
```
To train using Multi-processing Distributed DataParallel; follow the instructions in the official [PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).


### Results
The gradient norm ratios for ResNet, pre-act ResNet, and LeNo-ResNet over 200-layers depth network:

<img src="https://github.com/bharatmahaur/LeNo/blob/main/imgs/1.png" width="auto" height="225">
<img src="https://github.com/bharatmahaur/LeNo/blob/main/imgs/2.png" width="auto" height="225">
<img src="https://github.com/bharatmahaur/LeNo/blob/main/imgs/3.png" width="auto" height="225">

### Citation
If you use this code, please cite our paper:
```
@article{mahaur2022improved,
 title={Improved Residual Network Based on Norm-Preservation for Visual Recognition}, 
 author={Mahaur, Bharat and others},
 journal={Neural Networks},
 year={2022},
 publisher={Elsevier}
}
```
### Contact
Please contact [bharatmahaur@gmail.com](mailto:bharatmahaur@gmail.com) for any further queries.


### License
This code is released under the [Apache 2.0 License](LICENSE.md).
