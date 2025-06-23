source ~/.bashrc
source activate concat
export CUDA_DEVICE_ORDER=PCI_BUS_ID


# observation and prediction time settings:
# for twitter dataset, we use 3600*24*1 (86400, 1 day) or 3600*24*2 (172800, 2 days) as observation time
#                      we use 3600*24*32 (1382400, 16 days, 2764800) as prediction time
# for weibo   dataset, we use 1800 (0.5 hour) or 3600 (1 hour) as observation time
#                      we use 3600*24 (86400, 1 day) as prediction time
# for aps     dataset, we use 365*3 (1095, 3 years) or 365*5+1 (1826, 5 years) as observation time
#                      we use 365*20+5 (7305, 20 years) as prediction time
#--------------------------------------------------------------------------------------------------
#weibo
(
python gen_cas.py --data=data/weibo/ --observation_time=1800 --prediction_time=86400
python gen_emb.py --data=data/weibo/ --observation_time=1800 --max_seq=100
) &
(
python gen_cas.py --data=data/weibo/ --observation_time=3600 --prediction_time=86400
python gen_emb.py --data=data/weibo/ --observation_time=3600 --max_seq=100
) &

##--------------------------------------------------------------------------------------------------
#aps
#--------------------------------------------------------------------------------------------------

(
python gen_cas.py --data=data/aps/ --observation_time=1095 --prediction_time=7305
python gen_emb.py --data=data/aps/ --observation_time=1095 --max_seq=100
) &
(
python gen_cas.py --data=data/aps/ --observation_time=1826 --prediction_time=7305
python gen_emb.py --data=data/aps/ --observation_time=1826 --max_seq=100
) &

(
python gen_cas.py --data=data/twitter/ --observation_time=86400 --prediction_time=2764800
python gen_emb.py --data=data/twitter/ --observation_time=86400 --max_seq=100
) &
(
python gen_cas.py --data=data/twitter/ --observation_time=172800 --prediction_time=2764800
python gen_emb.py --data=data/twitter/ --observation_time=172800 --max_seq=100
) &
