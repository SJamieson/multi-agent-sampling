STARTDIR=$PWD
mkdir $1
cd $1
export PYTHONPATH="$STARTDIR:$PYTHONPATH"
python ../bayesopt_comparison.py 2.576 60 1 < /dev/null &
python ../bayesopt_comparison.py 2.576 60 61 < /dev/null &
python ../bayesopt_comparison.py 10.0 60 1 < /dev/null &
python ../bayesopt_comparison.py 10.0 60 61 < /dev/null &
python ../bayesopt_comparison.py 25.76 60 1 < /dev/null &
python ../bayesopt_comparison.py 25.76 60 61 < /dev/null &
python ../bayesopt_comparison.py 257.6 60 1 < /dev/null &
python ../bayesopt_comparison.py 257.6 60 61  < /dev/null &
wait
cd $STARTDIR
echo 'Done'
