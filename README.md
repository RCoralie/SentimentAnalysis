# back-DeepLearningMoviesProject
Python Back-end for the deep learning movies project 

You can find the project description [here](http://air.imag.fr/index.php/Suggestion_intelligente_de_films_bas%C3%A9e_sur_TensorFlow).

## Unit tests

You can run all unit tests by using 'python -m unittest discover' command anywhere on the path back-DeepLearningMoviesProject/MovieProject/tests/unit, the command will run scripts looking like 'test_something.py'.
To run only one test file, you just have to execute the script the classic way : 'python test_something.py'.

## Dependecies

### Liblinear

Liblinear is a library for Large Linear Classification, more informations [here](http://www.csie.ntu.edu.tw/~cjlin/liblinear)
To install it on python follows the command lines below:
```
$ wget http://www.csie.ntu.edu.tw/~cjlin/liblinear/liblinear-2.1.zip
$ unzip liblinear-2.1.zip 
$ cd liblinear-2.1/python
$ make
```

And add the folder "liblinear-2.1/python" into $PYTHONPATH

