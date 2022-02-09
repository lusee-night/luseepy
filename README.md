# luseepy
## About
A set of python utilities for performing various LuSEE Night related calculations.

## Packaging
Currently in the process of setting up packaging for PyPi, please stay tuned.

## Developing

To develop on your laptop, the easiest thing is to use the latest docker environment.
Please install docker and pull the image

```
docker pull uddhasystem/lusee-night-luseepy-jupyter:0.1 
```
Next, checkout the lusee repo
```
git clone git@github.com:lusee-night/luseepy.git
```

Next, define the following function for "lusee python" and "lusee jupyter"

```
lpython() { docker run  -e HOME -e PYTHONPATH=. -w $PWD -v $HOME:$HOME --user $(id -u):$(id -g) -it  buddhasystem/lusee-night-luseepy-jupyter:0.1  python $@; }
ljupyter() { port=9500; docker run  -e HOME -e PYTHONPATH=/path/to/luseepy -w $PWD -v $HOME:$HOME --user $(id -u):$(id -g) -itp $port:$port buddhasystem/lusee-night-luseepy-jupyter:0.1  /bin/bash -c "/usr/local/bin/jupyter notebook  --ip='*' --port=$port --no-browser  "; }
```

Inside your luseepy directory you can now run a test, for example, by saying:
```
lpython tests/LunarCalendarTest.py
```
To start jupyter, simply say
```
ljupyter
```
and then connect to the address given in the terminal output.



