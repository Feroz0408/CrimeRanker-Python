import pandas as pd
import got
import codecs
from pymongo import MongoClient
import csv
import time
import random
from flask import render_template
import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import nltk
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
import string
import numpy as np
from numpy._distributor_init import NUMPY_MKL
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from flask import Flask, redirect, url_for, request
