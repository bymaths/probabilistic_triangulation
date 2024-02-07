# Probabilistic Triangulation V2

Code of ICCV 2023 paper: [Probabilistic Triangulation for Uncalibrated Multi-View 3D Human Pose Estimation](https://arxiv.org/abs/2309.04756)

Abstract: 3D human pose estimation has been a long-standing challenge in computer vision and graphics, where multi-view methods have significantly progressed but are limited by the tedious calibration processes. Existing multi-view methods are restricted to fixed camera pose and therefore lack generalization ability. This paper presents a novel Probabilistic Triangulation module that can be embedded in a calibrated 3D human pose estimation method, generalizing it to uncalibration scenes. The key idea is to use a probability distribution to model the camera pose and iteratively update the distribution from 2D features instead of using camera pose. Specifically, We maintain a camera pose distribution and then iteratively update this distribution by computing the posterior probability of the camera pose through Monte Carlo sampling. This way, the gradients can be directly back-propagated from the 3D pose estimation to the 2D heatmap, enabling end-to-end training. Extensive experiments on Human3.6M and CMU Panoptic demonstrate that our method outperforms other uncalibration methods and achieves comparable results with state-of-the-art calibration methods. Thus, our method achieves a trade-off between estimation accuracy and generalizability.

## version update
1. Accelerated the model by replacing backbone with mobileone;
2. changed the sampling logic to speed up multi-view fusion;
3. Now the model can be reasoned in real time on iphone.

## Getting started

### 1. Dataset

Download and preprocess the dataset by following the instructions in [h36m-fetch](https://github.com/anibali/h36m-fetch) and [learnable triangulation](https://github.com/karfly/learnable-triangulation-pytorch).

The directory structure after completing all processing：

```
human3.6m
├── extra
│   ├── bboxes-Human36M-GT.npy
│   ├── human36m-multiview-labels-GTbboxes.npy
│   └── una-dinosauria-data
└── processed
    ├── S1
    ├── S11
    ├── S5
    ├── S6
    ├── S7
    ├── S8
    └── S9
```

### 2. Quick Start

Train the 2d backbone:

```python
python train2d.py
```

Train the 3d estimator, which by default will use the pre-trained model of the 2d backbone:
(pre-trained model will upload today)

```python
python train3d.py
```



## Citation

If you find this project useful for your research, please consider citing:

```
@article{hu2023pose,
  title={Probabilistic Triangulation for Uncalibrated Multi-View 3D Human Pose Estimation},
  author={Boyuan Jiang, Lei Hu, Shihong Xia}
  journal={IEEE International Conference on Computer Vision},
  year={2023},
  publisher={IEEE}
}
```
