# pyAFT
Python Airfoil Tools

## Mac:
* Download Xfoil binaries [here](https://drive.google.com/drive/folders/1eI0EObX7O90L_x9PwPydvI7Ko3O16kzN).

* Extract where you want and put the path to `xFoilPath` at around line 215 of ffdFoil.py

if you get an error that looks like this:
```
dyld: Library not loaded: /usr/local/opt/gcc/lib/gcc/9/libgfortran.5.dylib
  Referenced from: /Users/gaiera/Code/pyAFT/xfoil/Xfoil/./xfoil
  Reason: image not found
```

there might be some linking problems to your fortran libraries. you can try your luck at building from source, or just create a softlinks to your fortran library:

* Install gfortran (```brew install gfortran```)
* Create directory structure and softlinks as below:


create a soft link to where it is supposed to be:
```ln -s /usr/local/opt/gcc/lib/gcc/9/libgfortran.5.dylib /usr/local/gfortran/lib/libgfortran.5.dylib```

(```/usr/local/gfortran/lib/libgfortran.5.dylib``` is where it comes from brew, but ```locate libgfortran``` will give you some options you might have installed along the way with conda, scipy, or something like that)

(note: you might have to make the whole directory structure too)

You will have to do it for libquadmath too:
locate libquadmath.0.dylib (gives, /usr/local/gfortran/lib/libquadmath.0.dylib)
```ln -s /usr/local/gfortran/lib/libquadmath.0.dylib /usr/local/opt/gcc/lib/gcc/9/libquadmath.0.dylib```

* Now you should be able to go to where Xfoil is extracted and type `./xfoil` and it should run

