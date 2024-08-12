#!/bin/bash

cd data/method_holdout
for i in {0..39}
do
    if [ $i -gt 0 ]
	then
	       rm -r method$i
	       cp -r method0 method$i	
    fi
    cd method$i
    rm -r testset/*
    rm summary/*
    rm pkls/*
    rm tmp/*
    rm bins/*
    python3 prepare_fc_raw.py --holdout-method-index=$i --funcom=/nfs/projects/EyeContext/summaries/eyecontext_openai_summaries.pkl
    cd ..
    sleep 5
done
