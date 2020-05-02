# environment setup    
    source ~/.local/bin/virtualenvwrapper.sh
    source ~/.bashrc
    mkvirtualenv kaggle -p python3
    workon kaggle
    pip install numpy
    pip install tensorflow==2.0.0 
    pip install scikit-learn
    pip install matplotlib
    pip install spyder
    pip install Theano
    pip install keras
    pip install pandas
    pip install pillow
    
# Kaggle submission commands
    kaggle competitions submit -c titanic -f submission.csv -m "Message"