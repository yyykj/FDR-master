# FDR-master
<div align="center">
  <img src="https://github.com/yyykj/FDR-master/blob/main/fig.jpg">
</div>
  Mainframe diagram. The 3D point cloud data enters the farthest point sampling layer (FPS), grouping layer (Group), and Tucker decomposition layer
to get the core tensor features on the one hand, and the backbone network to get the classifier features on the other hand. The classifier features are aligned
with the feature dimensions through MLP and linear layers, and then distance features are calculated with the core tensor features. Finally, the distance features
are incorporated into the classifier features for classification.

  Taking pointmlp as an example (see mainly pointmlp.py), we perform TUCKER decomposition on the original data after FPS and Group operations to obtain the core tensor and factor matrix.
The core code is as follows:
  core, factors = tucker(X, rank=[a, b, c])
  The code of the tucker function is as follows：
```
def tucker(tensor, rank, fixed_factors=None, n_iter_max=100, init='svd',
           svd='numpy_svd', tol=10e-5, random_state=None, mask=None, verbose=False):
    if fixed_factors:
        try:
            (core, factors) = init
        except:
            raise ValueError(f'Got fixed_factor={fixed_factors} but no appropriate Tucker tensor was passed for "init".')
        fixed_factors = sorted(fixed_factors)
        modes_fixed, factors_fixed = zip(*[(i, f) for (i, f) in enumerate(factors) if i in fixed_factors])
        core = multi_mode_dot(core, factors_fixed, modes=modes_fixed)
        modes, factors = zip(*[(i, f) for (i, f) in enumerate(factors) if i not in fixed_factors])
        init = (core, list(factors))
        core, new_factors = partial_tucker(tensor, modes, rank=rank, n_iter_max=n_iter_max, init=init,
                                           svd=svd, tol=tol, random_state=random_state, mask=mask,
                                           verbose=verbose)
        factors = list(new_factors)
        for i, e in enumerate(fixed_factors):
            factors.insert(e, factors_fixed[i])
        core = multi_mode_dot(core, factors_fixed, modes=modes_fixed, transpose=True)
        return TuckerTensor((core, factors))
    else:
        modes = list(range(tl.ndim(tensor)))
        # TO-DO validate rank for partial tucker as well
        rank = validate_tucker_rank(tl.shape(tensor), rank=rank)
        core, factors = partial_tucker(tensor, modes, rank=rank, n_iter_max=n_iter_max, init=init,
                                       svd=svd, tol=tol, random_state=random_state, mask=mask,
                                       verbose=verbose)
        return TuckerTensor((core, factors))
```
  On the other hand, feature vectors passing through the pointmlp network are dimensionally aligned through the MLP and Linear layers. Then, the feature distance between two feature vectors is calculated by cosine_similarity() :
```output = torch.cosine_similarity(x, x2).CUDA () ```(The ablation experiment in this paper proves that the use of cosine similarity is more reasonable). 

  Finally, after some dimensionality processing, the distance feature is spliced with the feature vector output from the backbone network and input into the classification layer for object classification.

# Reference By
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

https://github.com/yanx27/Pointnet_Pointnet2_pytorch

https://github.com/ma-xu/pointMLP-pytorch

https://github.com/WangYueFt/dgcnn
