#!/bin/sh


cd ../
python sys_utterance.py -A extfea_clustering
python sys_utterance.py -A makefile
cd RL
python makeDA.py -A evaDA

