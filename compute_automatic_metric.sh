#!/bin/bash
learning_rate_list=(5e-5 3e-5 1e-5 )
epochs_list=(3 5 10)

#rm $FILENAME
#touch  $FILENAME
METEORFOLDER="metrics/leave-one-out/METEOR/gemini/pretrained/"
USEFOLDER="metrics/leave-one-out/USE/gemini/pretrained/"
PREDICTIONFILE="predictions/predict_context_pretrained_holdout_m"
## regular use score for method holdout 

for i in {0..39};do
	FILENAME=""
	FILENAME+=$USEFOLDER
	FILENAME+=summary$i.txt
	echo $FILENAME
	rm	$FILENAME
	touch	$FILENAME

	for j in {0..5};do
output=$((CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='0' python3 use_score_v.py $PREDICTIONFILE$i.txt --data=./data/method_holdout_context/context$i/summary --coms-filename=$j.test) 2>&1)
	echo  $output | tee -a $FILENAME
	done
done

## meteor score for method holdout 
for i in {0..39};do
        FILENAME=""
        FILENAME+=$METEORFOLDER
        FILENAME+=summary$i.txt
        echo $FILENAME
        rm      $FILENAME
        touch   $FILENAME

        for j in {0..5};do
output=$((python3 meteor.py $PREDICTIONFILE$i.txt --data=./data/method_holdout_context/context$i/summary --coms-filename=$j.test) 2>&1)
	#echo $output
        echo  $output | tee -a $FILENAME
        done
done











#for lr in ${learning_rate_list[@]};do
#        for epoch in ${epochs_list[@]}; do
#		output=$((CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='0' python3 contextuse/use_with_context.py --summary-filename=predictions/large_testset_freeze_lr$lr\_ep$epoch.txt) 2>&1)
#		PREDICTION_FILE="large_testset_freeze_lr$lr\\_ep$epoch.txt"
#		FINAL_STRING=""
#		FINAL_STRING+=$PREDICTION_FILE
#		FINAL_STRING+=$output
#		echo  $FINAL_STRING | tee -a $FILENAME
#		echo  | tee -a $FILENAME
#        done
#done



