# for a in $(seq 0 0.1 1.0); do
#       for k in $(seq 850 50 1250); do
#          python eval.py  --a $a --k $k
#        done
# done

python eval.py --threshold_h $threshold_h --threshold_l $threshold_l  --a $a --k $k  
