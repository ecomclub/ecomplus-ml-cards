# ecomplus-ml-cards
Python script for E-Com Plus Admin cards recomendation with machine learning and data science

# Technology stack
+ [Python 3](https://www.python.org/downloads/release/python-370/) 
+ [PIP](https://pypi.org/project/pip/)
+ [pandas](https://pandas.pydata.org/) 
+ [numpy](http://www.numpy.org/) 
+ [matplotlib](https://matplotlib.org/) 
+ [Surprise Lib](http://surpriselib.com/)

# Setting up
Installing dependencies on RHEL based Linux:

```bash
sudo yum install python27 python27-devel python-pip
pip install pandas
pip install numpy
pip install scikit-surprise
pip install matplotlib
```

## Running the script
```bash
python3 recommend-cards.py csv-input number-of-cards-result csv-output
```
