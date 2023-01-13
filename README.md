# Test-time Augmentation for Factual Probing

## environment setup

- GPU: CUDA 11.6
- miniconda4-4.7.12
  - (`pyenv install --list`) 2022.9.5 latest
- python3.9
 
 ## Dataset
 [This dataset](/v2.11d.csv) consists of 12500 unique facts from wikidata.
 Each fact has a subject, a relation, and an object.
 Facts are sampled with truncated sampling to reduce the biased distribution of the objects.
 The familiarity of a subject/object is controlled by the number of related facts in wikidata.
 It is also a disjoint set of [LAMA](https://github.com/facebookresearch/LAMA) trex

 ## Generate with TTA
 `python script/main.py`