export CUDA_VISIBLE_DEVICES=4

python train.py --epoch 50 --lr 0.001 --weight_decay 0.0005 --num_workers 64 --batch_size 64 --model_name resnet18

python train.py --epoch 50 --lr 0.001 --weight_decay 0.0005 --num_workers 64 --batch_size 64 --model_name resnet34

python train.py --epoch 50 --lr 0.001 --weight_decay 0.0005 --num_workers 64 --batch_size 64 --model_name resnet50

python train.py --epoch 50 --lr 0.001 --weight_decay 0.0005 --num_workers 64 --batch_size 64 --model_name resnet101

python train.py --epoch 50 --lr 0.001 --weight_decay 0.0005 --num_workers 64 --batch_size 64 --model_name resnet152
