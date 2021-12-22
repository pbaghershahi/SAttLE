
# Self-attention Presents Low-dimensional Knowledge Graph Embeddings for Link Prediction

The implementation of the paper in PyTorch will be publicly available soon!

## Link Prediction Results

**SAttLE results with TwoMult decoding**

Dataset | MRR | Hits@1 | Hits@3 | Hits@10
:--- | :---: | :---: | :---: | :---:
FB15k-237 | 0.352 | 0.261 | 0.386 | 0.535
WN18RR | 0.476 | 0.438 | 0.493 | 0.548

**SAttLE results with Tucker decoding**

Dataset | MRR | Hits@1 | Hits@3 | Hits@10
:--- | :---: | :---: | :---: | :---:
FB15k-237 | 0.349 | 0.258 | 0.382 | 0.533
WN18RR | 0.479 | 0.445 | 0.492 | 0.544

## Citation

Please cite the following paper if you find it useful:
```bash
@misc{baghershahi2021selfattention,
      title={Self-attention Presents Low-dimensional Knowledge Graph Embeddings for Link Prediction}, 
      author={Peyman Baghershahi and Reshad Hosseini and Hadi Moradi},
      year={2021},
      eprint={2112.10644},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
