# Excavator-pybullet

# Install conda

# Create the environment

```jsx
conda create -n pybullet_3_7 python=3.7
conda activate pybullet_3_7
```

# Install ruckig

`conda install ruckig` 报错：

```jsx
CMake Error at CMakeLists.txt:82 (find_package):
        By not providing "Findpybind11.cmake" in CMAKE_MODULE_PATH this project has
        asked CMake to find a package configuration file provided by "pybind11",
        but CMake did not find one.
      
        Could not find a package configuration file provided by "pybind11"
        (requested version 2.6) with any of the following names:
      
          pybind11Config.cmake
          pybind11-config.cmake
      
        Add the installation prefix of "pybind11" to CMAKE_PREFIX_PATH or set
        "pybind11_DIR" to a directory containing one of the above files.  If
        "pybind11" provides a separate development package or SDK, be sure it has
        been installed.

```

需要先安装：`conda install pybind11`

```jsx
conda install pybind11
pip install ruckig
pip install matplotlib==3.4.3
pip install numpy==1.21.3
pip install pybullet==3.2.0
```

# Configure URDF

```jsx
conda install pytorch torchvision -c pytorch
```

# Test

```cpp
python show_excavator.py
```

