# Deep 3d Portrait from a Single Image (CVPR2020)

## Presentation
- video: [Youtube link](https://youtu.be/t_XTpc22P8Y)
- pdf: [presentation.pdf](https://github.com/lylajeon/Deep3dPortrait/blob/main/presentation.pdf)
- final report: [report link](https://github.com/lylajeon/Deep3dPortrait/blob/main/Final_Report.pdf)

## Summary

This repostiory is forked from [sicxu/Deep3dPortrait](https://github.com/sicxu/Deep3dPortrait).

## Use custom image to generate 3d portrait objects of various expressions

```
git clone https://github.com/lylajeon/Deep3dPortrait Deep3dPortrait
cd Deep3dPortrait
git clone https://github.com/kingsj0405/Face-Landmark-Parsing face-parsing.PyTorch
cd ..
```
Put all files in outputs/step4 to step5_ui_expression&pose_change/result

```
python integrated_process.py --input_img [IMG_NAME]
cd step5_ui_expression&pose_change
chmod +x run.sh
./run.sh
```

# Origin README.md of Deep 3d Portrait from a Single Image (CVPR2020)
This is a tensorflow implementation of the following paper: [Deep 3d Portrait from a Single Image](https://arxiv.org/abs/2004.11598). We propose a two-step geometry learning scheme which first learn 3DMM face reconstruction from single images then learn to estimate hair and ear depth in a stereo setup.

## Getting Started
### System Requirements
- Software: Ubuntu 16.04, CUDA 9.0
- Python  >= 3.5
### Usage
1. Clone the repository
```
git clone https://github.com/sicxu/Deep3dPortrait.git
cd Deep3dPortrait
pip install -r requirements.txt
```
2. Follow the intructions in [Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction) to prepare the [BFM folder](/BFM)
3. Download the pretrained [face reconstruction model](https://drive.google.com/file/d/1fPsvLKghlCK8rknb9GPiKwIq9HIqWWwV/view?usp=sharing) and [depth estimation model](https://drive.google.com/file/d/1QUSK4k6ZONOZWpph9GHW0VGAiKFggEU5/view?usp=sharing), then put the pb files into the [model folder](model).
4. Run the following steps.

```
python step1_recon_3d_face.py
python step2_face_segmentation.py
python step3_get_head_geometry.py
python step4_save_obj.py
```

5. To check the results, see ./output subfolders which contain the results of corresponding steps.
   
## Others
- An image pre-alignment is necessary for face reconstruction. We recommend using [Bulat et al.'s method](https://github.com/1adrianb/2D-and-3D-face-alignment) to get facial landmarks (3D definition). We also need to use the masks of face,  hair and ear as input to the depth estimation network. We recommend using [Lin et al.'s method](https://arxiv.org/pdf/1906.01342.pdf) for semantic segmentation.
- The face reconstruction code is heavily borrowed from [Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction).
- The [render](utils/render) code is modified from [tf_mesh_render](https://github.com/google/tf_mesh_renderer/tree/ba27ea1798f6ee8d03ddbc52f42ab4241f9328bb). Note that the renderer we complied does not support other tensorflow versions and can only be used on Linux.
- The manipulation code will not be released. If you want to make a comparison with our method, please use the results in our paper, or you can contact me(sicheng_xu@yeah.net) for more comparisons.

## Citation
If you find this code helpful for your research, please cite our paper.