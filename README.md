# Domain Whitening Transform for Unsupervised Domain Adaptation (CVPR 2019)


Official PyTorch github repository for the paper [Unsupervised Domain Adaptation using Feature-Whitening and Consensus Loss](http://openaccess.thecvf.com/content_CVPR_2019/html/Roy_Unsupervised_Domain_Adaptation_Using_Feature-Whitening_and_Consensus_Loss_CVPR_2019_paper.html) published in The Conference on Computer Vision and Pattern Recognition (**CVPR**) held at Long Beach, California in June, 2019.

### Prerequisites
* Pytorch 1.0
* Python 3.5

### Usage
- Office-Home: To run the experiments on the [OfficeHome](http://hemanthdv.org/OfficeHome-Dataset/) dataset first you need to download the dataset from [this](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view) page. Following this step, you would need to download the ResNet50 pre-trained checkpoint, trained on ImageNet with the BatchNorm layers (in the first conv layer and the first Res block) replaced by *whitening* normalization layers. The pre-trained weights is available [here](https://drive.google.com/file/d/1Iw3pCXdiAiJJnZDzh7UToBNQipIVeMS2/view?usp=sharing).

  `$ python resnet50_dwt_mec_officehome.py --s_dset_path path-to-source-dataset-folder --t_dset_path path-to-target-dataset       folder --resnet_path path-to-pre-trained-resnet50-weights`
  
- USPS -> MNIST: 
```
python usps_mnist.py --group_size 4 --source 'usps' --target 'mnist'
```

If you find this code useful for your research, please cite our paper:
```
@article{roy2019unsupervised,
  title={Unsupervised Domain Adaptation using Feature-Whitening and Consensus Loss},
  author={Roy, Subhankar and Siarohin, Aliaksandr and Sangineto, Enver and Bulo, Samuel Rota and Sebe, Nicu and Ricci, Elisa},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```
