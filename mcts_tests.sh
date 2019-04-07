STARTDIR=$PWD
mkdir -p $1
cd $1
export PYTHONPATH="$STARTDIR:$PYTHONPATH"
#python ../mcts_comparison.py 2.576 5 2 < /dev/null &
python ../mcts_comparison.py 2.576 5 8 < /dev/null &
python ../mcts_comparison.py 10.0 5 8 < /dev/null &
#python ../mcts_comparison.py 25.76 5 2 < /dev/null &
python ../mcts_comparison.py 25.76 5 8 < /dev/null &
#python ../mcts_comparison.py 257.6 5 2 < /dev/null &
python ../mcts_comparison.py 257.6 5 8 < /dev/null &
wait
cd $STARTDIR
echo 'Done'
