for MODEL in vae
do
    for SEED in 1
    do
        mkdir -p checkpoints/$MODEL/$MODEL\_$SEED
        python scripts/run.py \
            --model $MODEL \
            --data data \
            --train_path data/train.csv \
            --test_path data/test.csv \
            --lr_n_restarts 5 \
            --save_frequency 10 \
            --checkpoint_dir checkpoints/$MODEL/$MODEL\_$SEED \
            --device cuda:$SEED \
            --metrics data/samples/$MODEL/metrics_$MODEL\_$SEED.csv \
            --seed $SEED \
            --gen_path data/samples/$MODEL/$MODEL\_$SEED.csv
    done
done
