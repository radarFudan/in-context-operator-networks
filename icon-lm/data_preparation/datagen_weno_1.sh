gpu=1

dir=data1208_weno_cubic_test
testeqns=11
testnum=100
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --eqn_types weno_cubic --dir $dir --name test --eqns $testeqns --num $testnum  --dt 0.0005 --file_split $testeqns --eqn_mode grid_-1_1 --truncate 10 --stride 5,10,20,50,100,150,200,250,300,400 --seed 101 &&

dir=data1208_weno_cubic_test_light
testeqns=5
testnum=100
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_weno.py --eqn_types weno_cubic --dir $dir --name test --eqns $testeqns --num $testnum  --dt 0.0005 --file_split $testeqns --eqn_mode grid_-1_1 --truncate 10 --stride 5,10,20,50,100,150,200,250,300,400 --seed 101 &&



echo "Done"

