#!/bin/sh
# Packaging and Release
docker run --workdir=$(pwd)/ --volume="/home/$USER:/home/$USER" tzutalin/py2qt4 /bin/sh -c 'make qt4py2; make test;sudo python setup.py sdist;sudo python setup.py install'

while true; do
    read -p "Do you wish to deploy this to PyPI(twine upload dist/* or pip install dist/*)?" yn
    case $yn in
        [Yy]* ) docker run -it --rm --workdir=$(pwd)/ --volume="/home/$USER:/home/$USER" tzutalin/py2qt4; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done
# python setup.py register
# python setup.py sdist upload
# Net pypi: twine upload dist/*

# Test before upladoing: pip install dist/labelImg.tar.gz
