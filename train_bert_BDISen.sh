time=$(date "+%Y%m%d-%H%M%S")
CUDA_VISIBLE_DEVICES=1 python bert.py  --mode "train" --model_name "/data1/lipengfei/basemodels/bert-base-uncased" --epochs 25 --batch_size 8 --max_length 256 --train_path "Dataset/train_BDISen.json" --val_path "Dataset/val_BDISen.json" --test_path "Dataset/test_BDISen.json" > logs/${time}Bert_BDISen.log