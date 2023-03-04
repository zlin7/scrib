# SCRIB: Set-classifier with Class-specific Risk Bounds

This repository contains the code for the AAAI 2022 paper "SCRIB: Set-classifier with Class-specific Risk Bounds for Blackbox Models" ([arxiv](https://arxiv.org/abs/2103.03945)).

Author List: Zhen Lin, Cao Xiao, Lucas Glass, M. Brandon Westover, Jimeng Sun.
 

# Demo
### Required Packages
SCRIB only deals with the base classifier's output. 
Thus, it only requires `numpy=1.19`.
It will also require `cython=0.29` to speed up the compute. 

### Example:
First, try `python -m opt.CoordDescent` to see if the cython code works properly.
`opt/CoordDescent.py` contains an example of how to use SCRIB.
  
We've also attached the predictions from the ISRUC, and you can run `demos/examples.ipynb` to replicate the experiment in our paper.
 




# Full Experiments
This includes codes that actually train the base classifier.
This part requires more packages and much longer time to process the data and run. 

### Required Packages
Packages we used to process data and train the deep learning models are:
`pytorch=1.7`,
`pandas=1.1.3`,
`mne=0.21.0`,
`scikit-learn=0.23.2`

Optional:
`pyunpack=0.2.2`


### Workflow
1. Specify the correct data paths in `_settings.py`, and follow `data.main.py` to process the data.  
2. run `dl_models/train.py` to train the models
3. run `dl_models/evaluate.py` to generate the dataframe we need for post-hoc processing.
4. Go back to the base example above.

 

 