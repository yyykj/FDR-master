# FDR-master
<div align="center">
  <img src="[https://github.com/yyykj/FDR-master/edit/main/fig.jpg](https://github.com/yyykj/FDR-master/blob/main/fig.jpg)">
</div>
Mainframe diagram. The 3D point cloud data enters the farthest point sampling layer (FPS), grouping layer (Group), and Tucker decomposition layer
to get the core tensor features on the one hand, and the backbone network to get the classifier features on the other hand. The classifier features are aligned
with the feature dimensions through MLP and linear layers, and then distance features are calculated with the core tensor features. Finally, the distance features
are incorporated into the classifier features for classification.


# Reference By
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

https://github.com/yanx27/Pointnet_Pointnet2_pytorch

https://github.com/ma-xu/pointMLP-pytorch

https://github.com/WangYueFt/dgcnn
