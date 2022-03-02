# ex: set ts=8 noet:

all: qt5 test

test: testpy3

testpy2:
	python -m unittest discover tests

testpy3:
	python3 -m unittest discover tests

qt4: qt4py2

qt5: qt5py3

qt4py2:
	pyrcc4 -py2 -o libs/resources.py resources.qrc

qt4py3:
	pyrcc4 -py3 -o libs/resources.py resources.qrc

qt5py3:
	pyrcc5 -o libs/resources.py resources.qrc

clean:
	rm -rf ~/.labelImgSettings.pkl *.pyc dist labelImg.egg-info __pycache__ build

pip_upload:
	python3 setup.py upload

long_description:
	restview --long-description

.PHONY: all
