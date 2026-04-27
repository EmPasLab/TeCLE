envs=(
    MiniGrid-KeyCorridorS3R3-v0
)
log_interval=10
save_interval=50
num_actors=16
total_frames=50000000
noisy_tv=(False)
seed=98
for noisy in ${noisy_tv[@]}; do
    for beta in ${noise_beta[@]}; do
        for env in ${envs[@]}; do
            model=Final_${algo}_rev_${env}_${frames}_noisy_tv_${noisy}_noise_beta_${beta}_seed_${seed}
            python3 -m scripts.train --algo $algo \
                                    --env $env \
                                    --log-interval $log_interval \
                                    --save-interval $save_interval \
                                    --num_actors $num_actors \
                                    --total_frames $total_frames \
                                    --model $model \
                                    --heatmap \
                                    --noisy_tv $noisy \
                                    --noise_beta $beta \
                                    --seed $seed \

        done
    done
done
