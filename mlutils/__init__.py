from .ema import *
from .utils import *
from .models import *
from .trainer import *
from .schedule import *
from .callbacks import *

# Set non-interactive backend globally
import matplotlib as mpl
mpl.use('agg')
