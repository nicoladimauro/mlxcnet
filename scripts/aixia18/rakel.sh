python3 runmeka.py Arts1500 -f 5 -c 26 -mc "meka.classifiers.multilabel.RAkEL  -M 10 -k 3 -P 0 -N 0 -S 0" -wc "weka.classifiers.functions.SMO" -o ./exp/meka/rakel/ > ./exp/meka/rakel/arts.log
python3 runmeka.py Business1500 -c 30 -f 5 -mc "meka.classifiers.multilabel.RAkEL  -M 10 -k 3 -P 0 -N 0 -S 0" -wc "weka.classifiers.functions.SMO" -o ./exp/meka/rakel/ > ./exp/meka/rakel/business.log
python3 runmeka.py CAL500 -c 174 -f 5 -mc "meka.classifiers.multilabel.RAkEL  -M 10 -k 3 -P 0 -N 0 -S 0" -wc "weka.classifiers.functions.SMO" -o ./exp/meka/rakel/ > ./exp/meka/rakel/cal.log
python3 runmeka.py emotions -c 6 -f 5 -mc "meka.classifiers.multilabel.RAkEL  -M 10 -k 3 -P 0 -N 0 -S 0" -wc "weka.classifiers.functions.SMO" -o ./exp/meka/rakel/ > ./exp/meka/rakel/emotions.log
python3 runmeka.py flags -c 7 -f 5 -mc "meka.classifiers.multilabel.RAkEL  -M 10 -k 3 -P 0 -N 0 -S 0" -wc "weka.classifiers.functions.SMO" -o ./exp/meka/rakel/ > ./exp/meka/rakel/flags.log
python3 runmeka.py Health1500 -c 32 -f 5 -mc "meka.classifiers.multilabel.RAkEL  -M 10 -k 3 -P 0 -N 0 -S 0" -wc "weka.classifiers.functions.SMO" -o ./exp/meka/rakel/ > ./exp/meka/rakel/health.log
python3 runmeka.py human3106 -c 14 -f 5 -mc "meka.classifiers.multilabel.RAkEL  -M 10 -k 3 -P 0 -N 0 -S 0" -wc "weka.classifiers.functions.SMO" -o ./exp/meka/rakel/ > ./exp/meka/rakel/human.log
python3 runmeka.py plant978 -c 12 -f 5 -mc "meka.classifiers.multilabel.RAkEL  -M 10 -k 3 -P 0 -N 0 -S 0" -wc "weka.classifiers.functions.SMO" -o ./exp/meka/rakel/ > ./exp/meka/rakel/plant.log
python3 runmeka.py scene -c 6 -f 5 -mc "meka.classifiers.multilabel.RAkEL  -M 10 -k 3 -P 0 -N 0 -S 0" -wc "weka.classifiers.functions.SMO" -o ./exp/meka/rakel/ > ./exp/meka/rakel/scene.log
python3 runmeka.py yeast -c 14 -f 5 -mc "meka.classifiers.multilabel.RAkEL  -M 10 -k 3 -P 0 -N 0 -S 0" -wc "weka.classifiers.functions.SMO" -o ./exp/meka/rakel/ > ./exp/meka/rakel/yeast.log

