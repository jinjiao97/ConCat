# ConCat
Xin Jing, Yichen Jing, Yuhuan Lu, Bangchao Deng, Sikun Yang, Dingqi Yang*, "
On Your Mark, Get Set, Predict! Modeling Continuous-Time Dynamics of Cascades
for Information Popularity Prediction". IEEE Transactions on Knowledge and Data Engineering 


## Basic Usage

### Environment

```shell
# create virtual environment
conda create --name concat python=3.7
# activate environment
conda activate concat
# install pytorch==1.12.0
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
# install other requirements
pip install -r requirements.txt
```

### Run the code
```shell
#Run at the root of this repo:
python setup.py build_ext --inplace
#preprocess
bash preprocess.sh
#train
bash run.sh

```


## Datasets

Datasets download link: [Google Drive](https://drive.google.com/file/d/1o4KAZs19fl4Qa5LUtdnmNy57gHa15AF-/view?usp=sharing) or [Baidu Drive (password: `1msd`)](https://pan.baidu.com/s/1tWcEefxoRHj002F0s9BCTQ).

The datasets we used in the paper are come from:

- [Twitter](http://carl.cs.indiana.edu/data/#virality2013) (Weng *et al.*, [Virality Prediction and Community Structure in Social Network](https://www.nature.com/articles/srep02522), Scientific Report, 2013).
- [Weibo](https://github.com/CaoQi92/DeepHawkes) (Cao *et al.*, [DeepHawkes: Bridging the Gap between 
Prediction and Understanding of Information Cascades](https://dl.acm.org/doi/10.1145/3132847.3132973), CIKM, 2017). You can also download Weibo dataset [here](https://drive.google.com/file/d/1fgkLeFRYQDQOKPujsmn61sGbJt6PaERF/view?usp=sharing) in Google Drive.  
- [APS](https://journals.aps.org/datasets) (Released by *American Physical Society*, obtained at Jan 17, 2019).  



