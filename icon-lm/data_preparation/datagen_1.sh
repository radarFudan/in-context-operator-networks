gpu=1
dir=data0910c_test_small
testeqns=100
testquests=1

CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --length 100 --dt 0.01 --eqn_types series_damped_oscillator --seed 101 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --length 100 --dx 0.01 --dt 0.02 --nu_nx_ratio 1 --eqn_types mfc_gparam_hj --seed 108 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --length 100 --dx 0.01 --dt 0.02 --nu_nx_ratio 1 --eqn_types mfc_rhoparam_hj --seed 109 &&

dir=data0910c
testeqns=100
testquests=5
traineqns=1000

CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --length 100 --dt 0.01 --eqn_types series_damped_oscillator --seed 101 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --length 100 --dx 0.01 --dt 0.02 --nu_nx_ratio 1 --eqn_types mfc_gparam_hj --seed 108 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --length 100 --dx 0.01 --dt 0.02 --nu_nx_ratio 1 --eqn_types mfc_rhoparam_hj --seed 109 &&

CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode train --name train --eqns $traineqns --length 100 --dt 0.01 --eqn_types series_damped_oscillator --seed 1 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode train --name train --eqns $traineqns --length 100 --dx 0.01 --dt 0.02 --nu_nx_ratio 1 --eqn_types mfc_gparam_hj --seed 8 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode train --name train --eqns $traineqns --length 100 --dx 0.01 --dt 0.02 --nu_nx_ratio 1 --eqn_types mfc_rhoparam_hj --seed 9 &&

echo "Done"



