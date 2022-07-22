
# Self-attention Presents Low-dimensional Knowledge Graph Embeddings for Link Prediction

This is a PyTorch implementation of the paper.

## Requirements

This implementation is in Python 3.6.6. Required packages are:

    numpy      1.15.1
    pytorch    1.0.1

## Usage

To train and evaluate SAttLE with TwoMult decoding for link prediction run the main.py script as follows:

For FB15k-237

python main.py -d fb15k-237 --gpu 0 --batch-size 2048 --evaluate-every 250 --n-epochs 1500 --lr 0.001 --lr-decay 0.995 --lr-step-decay 2 --n-layers 1 --d-embed 100 --num-head 64 --d-k 32 --d-v 50 --d-model 100 --d-inner 2048 --start-test-at 1000 --save-epochs 1100 --dr-enc 0.4 --dr-pff 0.2 --dr-sdp 0.1 --dr-mha 0.3 --decoder twomult

For WN18RR

python main.py -d wn18rr --gpu 0 --batch-size 1024 --evaluate-every 250 --n-epochs 4500 --lr 0.001 --lr-decay 1 --lr-step-decay 2 --n-layers 1 --d-embed 100 --num-head 64 --d-k 32 --d-v 50 --d-model 100 --d-inner 100 --start-test-at 5010 --save-epochs 6000 --dr-enc 0.3 --dr-pff 0.4 --dr-sdp 0.1 --dr-mha 0.4 --decoder twomult


To train with tucker decoder change "decoder" argument to "tucker" and set the other hyper-parameters based on the paper.

## Link Prediction Results

**SAttLE results with TwoMult decoding**

Dataset | MRR | Hits@1 | Hits@3 | Hits@10
:--- | :---: | :---: | :---: | :---:
FB15k-237 | 0.360 | 0.268 | 0.396 | 0.545
WN18RR | 0.491 | 0.454 | 0.508 | 0.558

**SAttLE results with Tucker decoding**

Dataset | MRR | Hits@1 | Hits@3 | Hits@10
:--- | :---: | :---: | :---: | :---:
FB15k-237 | 0.358 | 0.266 | 0.394 | 0.541
WN18RR | 0.476 | 0.442 | 0.490 | 0.540

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
