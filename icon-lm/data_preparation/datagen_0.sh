gpu=0
dir=data0910c_test_small
testeqns=100
testquests=1

CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --eqn_types ode_auto_const --seed 102 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --eqn_types ode_auto_linear1 --seed 103 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --eqn_types ode_auto_linear2 --seed 104 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --length 100 --dx 0.01 --eqn_types pde_poisson_spatial --seed 105 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --length 100 --dx 0.01 --eqn_types pde_porous_spatial --seed 106 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --length 100 --dx 0.01 --eqn_types pde_cubic_spatial --seed 107 &&

dir=data0910c
testeqns=100
testquests=5
traineqns=1000

CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --eqn_types ode_auto_const --seed 102 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --eqn_types ode_auto_linear1 --seed 103 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --eqn_types ode_auto_linear2 --seed 104 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --length 100 --dx 0.01 --eqn_types pde_poisson_spatial --seed 105 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --length 100 --dx 0.01 --eqn_types pde_porous_spatial --seed 106 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode test --name test --eqns $testeqns --quests $testquests --length 100 --dx 0.01 --eqn_types pde_cubic_spatial --seed 107 &&

CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode train --name train --eqns $traineqns --eqn_types ode_auto_const --seed 2 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode train --name train --eqns $traineqns --eqn_types ode_auto_linear1 --seed 3 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode train --name train --eqns $traineqns --eqn_types ode_auto_linear2 --seed 4 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode train --name train --eqns $traineqns --length 100 --dx 0.01 --eqn_types pde_poisson_spatial --seed 5 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode train --name train --eqns $traineqns --length 100 --dx 0.01 --eqn_types pde_porous_spatial --seed 6 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen.py --dir $dir --caption_mode train --name train --eqns $traineqns --length 100 --dx 0.01 --eqn_types pde_cubic_spatial --seed 7 &&

echo "Done"



