python3 runmeka.py Arts1500 -f 5 -c 26 -mc "meka.classifiers.multilabel.CC -S 0" -wc "weka.classifiers.trees.J48 -- -C 0.25 -M 2" -o ./exp/meka/cc/ > ./exp/meka/cc/arts.log
python3 runmeka.py Business1500 -c 30 -f 5 -mc "meka.classifiers.multilabel.CC -S 0" -wc "weka.classifiers.trees.J48 -- -C 0.25 -M 2" -o ./exp/meka/cc/ > ./exp/meka/cc/business.log
python3 runmeka.py CAL500 -c 174 -f 5 -mc "meka.classifiers.multilabel.CC -S 0" -wc "weka.classifiers.trees.J48 -- -C 0.25 -M 2" -o ./exp/meka/cc/ > ./exp/meka/cc/cal.log
python3 runmeka.py emotions -c 6 -f 5 -mc "meka.classifiers.multilabel.CC -S 0" -wc "weka.classifiers.trees.J48 -- -C 0.25 -M 2" -o ./exp/meka/cc/ > ./exp/meka/cc/emotions.log
python3 runmeka.py flags -c 7 -f 5 -mc "meka.classifiers.multilabel.CC -S 0" -wc "weka.classifiers.trees.J48 -- -C 0.25 -M 2" -o ./exp/meka/cc/ > ./exp/meka/cc/flags.log
python3 runmeka.py Health1500 -c 32 -f 5 -mc "meka.classifiers.multilabel.CC -S 0" -wc "weka.classifiers.trees.J48 -- -C 0.25 -M 2" -o ./exp/meka/cc/ > ./exp/meka/cc/health.log
python3 runmeka.py human3106 -c 14 -f 5 -mc "meka.classifiers.multilabel.CC -S 0" -wc "weka.classifiers.trees.J48 -- -C 0.25 -M 2" -o ./exp/meka/cc/ > ./exp/meka/cc/human.log
python3 runmeka.py plant978 -c 12 -f 5 -mc "meka.classifiers.multilabel.CC -S 0" -wc "weka.classifiers.trees.J48 -- -C 0.25 -M 2" -o ./exp/meka/cc/ > ./exp/meka/cc/plant.log
python3 runmeka.py scene -c 6 -f 5 -mc "meka.classifiers.multilabel.CC -S 0" -wc "weka.classifiers.trees.J48 -- -C 0.25 -M 2" -o ./exp/meka/cc/ > ./exp/meka/cc/scene.log
python3 runmeka.py yeast -c 14 -f 5 -mc "meka.classifiers.multilabel.CC -S 0" -wc "weka.classifiers.trees.J48 -- -C 0.25 -M 2" -o ./exp/meka/cc/ > ./exp/meka/cc/yeast.log

