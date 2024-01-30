gpu=0

# dir=data0829_weno_quadratic
# testeqns=10
# testquests=1
# traineqns=100
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --dir $dir --name test --eqns $testeqns --quests $testquests --file_split 1 --seed 101 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --dir $dir --name train --eqns $traineqns --quests 1 --file_split 10 --seed 1 &&

# dir=data0831_weno_quadratic_test
# testeqns=21
# testnum=500
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --dir $dir --name test --eqns $testeqns --num $testnum --file_split $testeqns --eqn_mode grid_-1_1 --truncate 10 --seed 101 &&

# dir=data0831_weno_quadratic_test_light
# testeqns=11
# testnum=100
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --dir $dir --name test --eqns $testeqns --num $testnum --file_split $testeqns --eqn_mode grid_-1_1 --truncate 10 --seed 101 &&

# dir=data0831_weno_quadratic_ood
# testeqns=21
# testnum=500
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --dir $dir --name test --eqns $testeqns --num $testnum --file_split $testeqns --eqn_mode grid_-2_2 --truncate 10 --seed 101 &&


# dir=data0904_weno_cubic
# traineqns=1000
# trainnum=100
# testeqns=10
# testnum=100
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --eqn_types weno_cubic --dir $dir --name test --eqns $testeqns --num $testnum --dt 0.0005 --file_split 1  --truncate 100 --seed 101 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --eqn_types weno_cubic --dir $dir --name train --eqns $traineqns --num $trainnum --dt 0.0005 --file_split 10 --truncate 100 --seed 1 &&

# dir=data0904_weno_cubic_test
# testeqns=11
# testnum=100
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --eqn_types weno_cubic --dir $dir --name test --eqns $testeqns --num $testnum  --dt 0.0005 --file_split $testeqns --eqn_mode grid_-1_1 --truncate 10 --seed 101 &&

# dir=data0904_weno_cubic_test_light
# testeqns=5
# testnum=100
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --eqn_types weno_cubic --dir $dir --name test --eqns $testeqns --num $testnum  --dt 0.0005 --file_split $testeqns --eqn_mode grid_-1_1 --truncate 10 --seed 101 &&

# instable
# dir=data0904_weno_cubic_ood
# testeqns=11
# testnum=100
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --eqn_types weno_cubic --dir $dir --name test --eqns $testeqns --num $testnum  --dt 0.0005 --file_split $testeqns --eqn_mode grid_-2_2 --truncate 10 --seed 101 &&


# dir=data1208_weno_cubic
# traineqns=1000
# trainnum=100
# testeqns=10
# testnum=100
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --eqn_types weno_cubic --dir $dir --name test --eqns $testeqns --num $testnum --dt 0.0005 --file_split 1  --truncate 100 --stride 20,40,60,80,100,120,140,160,180,200 --seed 101 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --eqn_types weno_cubic --dir $dir --name train --eqns $traineqns --num $trainnum --dt 0.0005 --file_split 10 --truncate 100 --stride 20,40,60,80,100,120,140,160,180,200 --seed 1 &&

# dir=data1208_weno_cubic_test
# testeqns=11
# testnum=100
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --eqn_types weno_cubic --dir $dir --name test --eqns $testeqns --num $testnum  --dt 0.0005 --file_split $testeqns --eqn_mode grid_-1_1 --truncate 10 --stride 5,10,20,50,100,150,200,250,300,400 --seed 101 &&

# dir=data1208_weno_cubic_test_light
# testeqns=5
# testnum=100
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --eqn_types weno_cubic --dir $dir --name test --eqns $testeqns --num $testnum  --dt 0.0005 --file_split $testeqns --eqn_mode grid_-1_1 --truncate 10 --stride 5,10,20,50,100,150,200,250,300,400 --seed 101 &&

dir=data1209_weno_sin_fix
testeqns=1
testquests=10
testnum=100
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --eqn_types weno_sin --dir $dir --name test --eqns $testeqns --num $testnum --quests $testquests --dt 0.0005 --file_split $testeqns --eqn_mode fix_1_-1_1 --truncate 10 --stride 5,10,20,50,100,150,200,250,300,400 --seed 1001 &&


echo "Done"

