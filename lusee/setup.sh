dir=`pwd`
parentdir="$(dirname "$dir")"
echo $parentdir
export PYTHONPATH=$PYTHONPATH:$parentdir
echo PYTHONPATH: $PYTHONPATH

