STARTDIR=$PWD
mkdir -p $1
cd $1
export PYTHONPATH="$STARTDIR:$PYTHONPATH"
python ../serial_mcts.py 2.576 5 4 90 < /dev/null &
python ../serial_mcts.py 10.0 5 4 90 < /dev/null &
python ../serial_mcts.py 25.76 5 4 90 < /dev/null &
python ../serial_mcts.py 257.6 5 4 90 < /dev/null &
#wait
python ../serial_mcts.py 2.576 5 8 60 < /dev/null &
python ../serial_mcts.py 10.0 5 8 60 < /dev/null &
python ../serial_mcts.py 25.76 5 8 60 < /dev/null &
python ../serial_mcts.py 257.6 5 8 60 < /dev/null &
wait
cd $STARTDIR
echo 'Done'
