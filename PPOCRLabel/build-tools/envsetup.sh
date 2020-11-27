#!/bin/sh

THIS_SCRIPT_PATH=`readlink -f $0`
THIS_SCRIPT_DIR=`dirname ${THIS_SCRIPT_PATH}`
#OS Ubuntu 14.04
### Common packages for linux/windows
if [ ! -e "pyinstaller" ]; then
    git clone https://github.com/pyinstaller/pyinstaller
    cd pyinstaller
    git checkout v2.1 -b v2.1
    cd ${THIS_SCRIPT_DIR}
fi

echo "Going to clone and download packages for building windows"
#Pacakges
#>  pyinstaller (2.1)
#>  wine (1.6.2)
#>  virtual-wine (0.1)
#>  python-2.7.8.msi
#>  pywin32-218.win32-py2.7.exe

## tool to install on Ubuntu
#$ sudo apt-get install wine

### Clone a repo to create virtual wine env
if [ ! -e "virtual-wine" ]; then
	git clone https://github.com/htgoebel/virtual-wine.git
fi

apt-get install scons
### Create virtual env
rm -rf venv_wine
./virtual-wine/vwine-setup venv_wine
#### Active virutal env
. venv_wine/bin/activate

### Use wine to install packages to virtual env
if [ ! -e "python-2.7.8.msi" ]; then
    wget "https://www.python.org/ftp/python/2.7.8/python-2.7.8.msi"
fi

if [ ! -e "pywin32-218.win32-py2.7.exe" ]; then
    wget "http://nchc.dl.sourceforge.net/project/pywin32/pywin32/Build%20218/pywin32-218.win32-py2.7.exe"
fi

if [ ! -e "PyQt4-4.11.4-gpl-Py2.7-Qt4.8.7-x32.exe" ]; then
    wget "http://nchc.dl.sourceforge.net/project/pyqt/PyQt4/PyQt-4.11.4/PyQt4-4.11.4-gpl-Py2.7-Qt4.8.7-x32.exe"
fi

if [ ! -e "lxml-3.7.3.win32-py2.7.exe" ]; then
    wget "https://pypi.python.org/packages/a3/f6/a28c5cf63873f6c55a3eb7857b736379229b85ba918261d2e88cf886905e/lxml-3.7.3.win32-py2.7.exe#md5=a0f746355876aca4ca5371cb0f1d13ce"
fi

