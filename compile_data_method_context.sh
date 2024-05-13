#!/bin/bash

cd data/method_holdout_context
for i in {0..39}
do
    if [ $i -gt 0 ]
	then
	       rm -r context$i
	       cp -r context0 context$i	
    fi
    cd context$i
    rm -r testset/*
    rm summary/*
    rm pkls/*
    rm tmp/*
    rm bins/*
    python3 prepare_fc_raw.py --holdout-method-index=$i --funcom=/nfs/projects/EyeContext/summaries/eyecontext_openai_summaries.pkl
    python3 testdatagen.py --holdout-method-index=$i
    cd ..
done
