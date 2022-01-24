import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.axes as maxes
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

from customscripts import utils
from customscripts import configuration

formatter = ticker.FuncFormatter(utils.human_format)
