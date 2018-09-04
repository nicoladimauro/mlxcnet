mkdir exp/mlcsn/d0.1cnet/
python3 runmlcsn.py Arts1500 -c 26 -l -d 0.1 -o ./exp/mlcsn/d0.1cnet/ > ./exp/mlcsn/d0.1cnet/arts.log
python3 runmlcsn.py Business1500 -c 30 -l -d 0.1  -o ./exp/mlcsn/d0.1cnet/ > ./exp/mlcsn/d0.1cnet/business.log
python3 runmlcsn.py CAL500 -c 174 -l -d 0.1  -o ./exp/mlcsn/d0.1cnet/ > ./exp/mlcsn/d0.1cnet/cal.log
python3 runmlcsn.py emotions -c 6 -l -d 0.1  -o ./exp/mlcsn/d0.1cnet/ > ./exp/mlcsn/d0.1cnet/emotions.log
python3 runmlcsn.py flags -c 7 -l -d 0.1  -o ./exp/mlcsn/d0.1cnet/ > ./exp/mlcsn/d0.1cnet/flags.log
python3 runmlcsn.py Health1500 -c 32 -l -d 0.1  -o ./exp/mlcsn/d0.1cnet/ > ./exp/mlcsn/d0.1cnet/health.log
python3 runmlcsn.py human3106 -c 14 -l -d 0.1  -o ./exp/mlcsn/d0.1cnet/ > ./exp/mlcsn/d0.1cnet/human.log
python3 runmlcsn.py plant978 -c 12 -l -d 0.1  -o ./exp/mlcsn/d0.1cnet/ > ./exp/mlcsn/d0.1cnet/plant.log
python3 runmlcsn.py scene -c 6 -l -d 0.1  -o ./exp/mlcsn/d0.1cnet/ > ./exp/mlcsn/d0.1cnet/scene.log
python3 runmlcsn.py yeast -c 14 -l -d 0.1  -o ./exp/mlcsn/d0.1cnet/ > ./exp/mlcsn/d0.1cnet/yeast.log

mkdir exp/mlcsn/d0.1xcnetc11/
python3 runmlcsn.py Arts1500 -c 26 -l -d 0.1 -x -k 11 -o ./exp/mlcsn/d0.1xcnetc11/ > ./exp/mlcsn/d0.1xcnetc11/arts.log
python3 runmlcsn.py Business1500 -c 30 -l -d 0.1  -x -k 11 -o ./exp/mlcsn/d0.1xcnetc11/ > ./exp/mlcsn/d0.1xcnetc11/business.log
python3 runmlcsn.py CAL500 -c 174 -l -d 0.1  -x -k 11 -o ./exp/mlcsn/d0.1xcnetc11/ > ./exp/mlcsn/d0.1xcnetc11/cal.log
python3 runmlcsn.py emotions -c 6 -l -d 0.1  -x -k 11 -o ./exp/mlcsn/d0.1xcnetc11/ > ./exp/mlcsn/d0.1xcnetc11/emotions.log
python3 runmlcsn.py flags -c 7 -l -d 0.1  -x -k 11 -o ./exp/mlcsn/d0.1xcnetc11/ > ./exp/mlcsn/d0.1xcnetc11/flags.log
python3 runmlcsn.py Health1500 -c 32 -l -d 0.1  -x -k 11 -o ./exp/mlcsn/d0.1xcnetc11/ > ./exp/mlcsn/d0.1xcnetc11/health.log
python3 runmlcsn.py human3106 -c 14 -l -d 0.1  -x -k 11 -o ./exp/mlcsn/d0.1xcnetc11/ > ./exp/mlcsn/d0.1xcnetc11/human.log
python3 runmlcsn.py plant978 -c 12 -l -d 0.1  -x -k 11 -o ./exp/mlcsn/d0.1xcnetc11/ > ./exp/mlcsn/d0.1xcnetc11/plant.log
python3 runmlcsn.py scene -c 6 -l -d 0.1  -x -k 11 -o ./exp/mlcsn/d0.1xcnetc11/ > ./exp/mlcsn/d0.1xcnetc11/scene.log
python3 runmlcsn.py yeast -c 14 -l -d 0.1  -x -k 11 -o ./exp/mlcsn/d0.1xcnetc11/ > ./exp/mlcsn/d0.1xcnetc11/yeast.log
