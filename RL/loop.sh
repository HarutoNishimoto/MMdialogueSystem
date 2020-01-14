#!/bin/sh


#for (( i = 10 ; i < 40 ; i=i+10 ));
for i in 30 50
do
	for j in 1 5 10 50
	do
	#python q_learning.py -A chk --ep 500 --R_oneUI 10 --R_persistUI ${i} --Rc_bigram ${j} --model 17_r2-10-${i}-${j}c
	python q_learning.py -A train --ep 500 --R_oneUI 10 --R_persistUI ${i} --Rc_bigram ${j} --model 17_r3-10-${i}-${j}c > 17_r3-10-${i}-${j}c
	done
done

