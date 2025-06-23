# observation and prediction time settings:
# for aps     dataset, we use 365*3 (1095, 3 years) or 365*5+1 (1826, 5 years)
#                      we use 365*20+5 (7305, 20 years) as prediction time
# for weibo   dataset, we use 1800 (0.5 hour) or 3600 (1 hour) as observation time
#                      we use 3600*24 (86400, 1 day) as prediction time
# for twitter dataset, we use 3600*24*1 (86400, 1 day) or 3600*24*2 (172800, 2 days) as observation time
#                      we use 3600*24*32 (2764800, 32 days) as prediction time

#--------------------------------------------------------------------------------------------------

### APS
# APS -3y -seq 100
python train.py  --data=aps --observation_time=1095 --max_seq=100 --max_events=20000 --tpp_hdims=64
# APS -5y -seq 100
python train.py  --data=aps --observation_time=1826 --max_seq=100 --max_events=10000 --tpp_hdims=64


### Weibo
# Weibo -0.5h -seq 100
python train.py  --data=weibo --observation_time=1800 --max_seq=100 --max_events=10000 --tpp_hdims=64
# Weibo -1h -seq 100
python train.py  --data=weibo --observation_time=3600 --max_seq=100 --max_events=40000 --tpp_hdims=128 -lr=0.0015


### Twitter
# Twitter -1d -seq 100
python train.py  --data=twitter --observation_time=86400 --max_seq=100 --max_events=10000 --tpp_hdims=32
# Twitter -2d -seq 100
python train.py  --data=twitter --observation_time=172800 --max_seq=100 --max_events=10000 --tpp_hdims=16
