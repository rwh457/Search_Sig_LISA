export CUDA_VISIBLE_DEVICES=0
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1


python3.7 ../model/main.py train new \
    --data_dir ../data \
    --model_dir ../output/demo_model \
    --save_model_name model.pt \
    --data.z 1 \
    --data.target_percentage 0.5 0.8 \
    --data.epoch_size 200 \
    --data.batch_size 64 \
    --data.num_workers 0 \
    --train.num_epochs 5 \
    --train.lr 0.00001 \
    --train.no_lr_annealing \
    --train.lr_anneal_method cosine \
    --train.output_freq 1 \

# python ../model/main.py train existing \
#     --data_dir ../data \
#     --model_dir ../output/demo_model \
#     --save_model_name model.pt \
#     --data.z 2 \
#     --data.epoch_size 64 \
#     --data.batch_size 2 \
#     --data.num_workers 0 \
#     --train.num_epochs 5 \
#     --train.lr 0.0001 \
#     --train.no_lr_annealing \    
#     --train.lr_anneal_method cosine \
#     --train.output_freq 5 \
