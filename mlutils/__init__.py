from .ema import *
from .utils import *
from .models import *
from .trainer import *
from .schedule import *
from .callbacks import *
from .run_timer import RunTimer, disabled_timer
from .run_log import close_run_log, log_config_trace, log_run_banner, setup_run_log
from .metrics import format_metric, format_scalar, metric_is_set, normalize_metric, unset_metric

# Set non-interactive backend globally
import matplotlib as mpl
mpl.use('agg')
