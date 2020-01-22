#!/bin/sh


#for (( i = 10 ; i < 40 ; i=i+10 ));
for i in 99 995
do
	for j in 1 2
	do
		python q_learning.py -A train --model 200117_c0${i}_s${j} --coef_epsilon 0.${i} --seed ${j}
		python forRL.py -A Qval -I 200117_c0${i}_s${j} -T Q
		python forRL.py -A Qval -I 200117_c0${i}_s${j} -T Qfreq
	done
done

