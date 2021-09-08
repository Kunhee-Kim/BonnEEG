
for i in 7 10 13 16
do
  for j in 32 64 128 256
  do
    for k in 2
    do
      python main.py --training_epochs $i --layer_features $j 3 --layers $k
      done

    for k in 3
    do
      for l in 64
      do
        python main.py --training_epochs $i --layer_features $j $l 3 --layers $k
      done
    done
  done
done

