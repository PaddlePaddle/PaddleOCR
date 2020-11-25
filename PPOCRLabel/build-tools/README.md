### Deploy to PyPI

```
cd [ROOT]
sh build-tools/build-for-pypi.sh
```

### Build for Ubuntu

```
cd build-tools
sh run-in-container.sh
sh envsetup.sh
sh build-ubuntu-binary.sh
```

### Build for Windows

```
cd build-tools
sh run-in-container.sh
sh envsetup.sh
sh build-windows-binary.sh
```

### Build for macOS High Sierra
```
cd build-tools
./build-for-macos.sh
```

Note: If there are some problems, try to
```
sudo rm -rf virtual-wne venv_wine
```
