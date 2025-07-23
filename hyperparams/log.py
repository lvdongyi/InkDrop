import contextlib
import functools
import os
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

import threading
import time
import logging
import colorlog
from hyperparams.general_params import general_args
import wandb
log_config = {
    "DEBUG": {"level": 10, "color": "purple"},
    "INFO": {"level": 20, "color": "green"},
    "TRAIN": {"level": 21, "color": "cyan"},
    "EVAL": {"level": 22, "color": "blue"},
    "WARNING": {"level": 30, "color": "yellow"},
    "ERROR": {"level": 40, "color": "red"},
    "CRITICAL": {"level": 50, "color": "bold_red"},
}

class Logger(object):
    def __init__(self, name: str = None, log_file: str = None):
        name = "FedDoge" if not name else name
        self.logger = logging.getLogger(name)

        # 记录脚本启动时间并格式化为 MMDDhhmmss
        st = time.localtime(time.time())
        self.start_time = st
        self.start_time_str = (
            f"#{st.tm_mon:02}"
            f"{st.tm_mday:02}"
            f"{st.tm_hour:02}"
            f"{st.tm_min:02}"
            f"{st.tm_sec:02}#"
        )
        self.start_time_str_wo_pound = (
            f"{st.tm_mon:02}"
            f"{st.tm_mday:02}"
            f"{st.tm_hour:02}"
            f"{st.tm_min:02}"
            f"{st.tm_sec:02}"
        )

        # 添加自定义 log levels
        for key, conf in log_config.items():
            logging.addLevelName(conf["level"], key)
            self.__dict__[key]       = functools.partial(self.__call__, conf["level"])
            self.__dict__[key.lower()] = functools.partial(self.__call__, conf["level"])

        # 控制台 Formatter：在 asctime 之前插入 %(start_time)s
        console_fmt = "%(log_color)s%(start_time)s [%(asctime)-15s] [%(levelname)8s]%(reset)s - %(message)s"
        self.console_formatter = colorlog.ColoredFormatter(
            console_fmt,
            log_colors={k: v["color"] for k, v in log_config.items()},
        )

        # 控制台 Handler
        self.handler = logging.StreamHandler()
        self.handler.setFormatter(self.console_formatter)

        # 文件 Handler（如果需要，也可以把 start_time 加进文件里）
        log_file = log_file if log_file else os.path.join(
            f'../results/{general_args.synthesis_method.lower()}/{general_args.dataset.lower()}/',
            f"{self.start_time_str_wo_pound}",
            "autolog.log"
        )
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            self.file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_fmt = "%(start_time)s %(asctime)s [%(levelname)-8s] - %(message)s"
            file_formatter = logging.Formatter(file_fmt)
            self.file_handler.setFormatter(file_formatter)
            self.logger.addHandler(self.file_handler)

        # 把控制台 handler 加进 logger
        self.logger.addHandler(self.handler)

        # 默认 DEBUG
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self._is_enable = True

    def disable(self):
        self._is_enable = False

    def enable(self):
        self._is_enable = True

    def set_level(self, log_level: str):
        assert log_level in log_config, f"Invalid log level. Choose among {log_config.keys()}"
        self.logger.setLevel(log_level)

    @property
    def is_enable(self) -> bool:
        return self._is_enable

    def __call__(self, log_level: int, msg: str):
        if not self.is_enable:
            return
        # wandb 里只存原始消息
        wandb.log({str(log_level): str(msg)})
        # 这里通过 extra 把 start_time_str 注入到 Formatter
        self.logger.log(log_level, msg, extra={"start_time": self.start_time_str})

    @contextlib.contextmanager
    def use_terminator(self, terminator: str):
        old = self.handler.terminator
        self.handler.terminator = terminator
        yield
        self.handler.terminator = old

    @contextlib.contextmanager
    def processing(self, msg: str, interval: float = 0.1):
        end = False
        def _printer():
            idx = 0
            flags = ["\\", "|", "/", "-"]
            while not end:
                flag = flags[idx % len(flags)]
                with self.use_terminator("\r"):
                    self.info(f"{msg}: {flag}")
                time.sleep(interval)
                idx += 1
        t = threading.Thread(target=_printer)
        t.start()
        yield
        end = True

    @functools.lru_cache(None)
    def warning_once(self, *args, **kwargs):
        self.warning(*args, **kwargs)

    @functools.lru_cache(None)
    def info_once(self, *args, **kwargs):
        self.info(*args, **kwargs)


logger = Logger()
wandb.init(project="FedDOGE", config=general_args, dir="../results")

start_time = logger.start_time_str