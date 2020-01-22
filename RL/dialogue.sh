#!/bin/sh


# sh dialogue.sh [NUM]で指定する
# NUM=1はベース1
# NUM=2はベース2
# NUM=3は提案


if [ "$1" = "1" ] ; then
	python q_learning.py -A dialogue_Q

elif [ "$1" = "2" ] ; then
	python q_learning.py -A dialogue --model 200110_baseDA

elif [ "$1" = "3" ] ; then
	python q_learning.py -A dialogue --model 200109

elif [ "$1" = "rewrite" ] ; then
	python forRL.py -A rewrite

else
	echo "rewrite 1-3."
fi