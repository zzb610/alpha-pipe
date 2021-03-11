from abc import ABCMeta
import traceback
from multiprocessing import Pool
import time
import datetime as dt
from loguru import logger


class BaseParallel(object, metaclass=ABCMeta):
    def __init__(self, processes):
        """并行基类

        Parameters
        ----------
        processes : optional
            并行数量, by default cpu_count()
        """
        self.total_processes = 0
        self.completed_processes = 0
        self.results = []
        self.data = []
        self.cores = processes  # cpu核心数量
        self._loginfolist = []  # 保存打印信息

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        print(self.__dict__)
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        print(state)
        self.__dict__.update(state)

    def get_results(self):
        return self.results

    def complete(self, result):
        self.results.extend(result)
        self.completed_processes += 1


class Parallel(BaseParallel):
    """多进程类
    """

    def __init__(self, processes, initializer, init_args):
        super(Parallel, self).__init__(processes)
        self.pool = Pool(processes, initializer, init_args)

    def run(self, func, iter):
        if isinstance(iter, list) and self.cores > 1 and len(iter) >= self.cores:
            n_cores = self.cores
            for i in range(n_cores):
                n_per_core = int(len(iter) / n_cores) + 1  # 每个进程运行的数据数量
                self.data.append(
                    self.pool.starmap_async(func, iter[i * n_per_core:(i + 1) * n_per_core], callback=self.complete, error_callback=self.exception))
                self.total_processes += 1
        else:
            self.data.append(self.pool.starmap_async(
                func=func, iterable=iter, callback=self.complete, error_callback=self.exception))
            self.total_processes += 1
        for i in range(self.total_processes):
            try:
                while not self.data[i].ready():
                    time.sleep(0.5)
                self.data[i].get()
            except Exception as e:
                msg = str(e)
                if "JQData02" in msg:
                    now = dt.datetime.today()
                    next = (now + dt.timedelta(hours=2))
                    sleep_time = (next - now).total_seconds()
                    logger.info('等待数据权限更新...下次启动时间: {}'.format(next))
                    time.sleep(sleep_time)
                logger.exception(e)
        self.pool.close()
        self.pool.join()

    def exception(self, exception=None):
        logger.exception(exception)


# class Parallel_Thread(BaseParallel):
#     """ 多线程类
#     """

#     def __init__(self, processes=cpu_count()):
#         super(Parallel_Thread, self).__init__(processes)
#         self.pool = ThreadPoolExecutor(self.cores)

#     def run(self, iter):
#         if isinstance(iter, list) and self.cores > 1 and len(iter) > self.cores:
#             n_core = self.cores
#             for i in range(n_core):
#                 n_per = int(len(iter) / n_core) + 1
#                 self.data.append(self.pool.map(
#                     self.do_working, iter[i * n_per:(i + 1) * n_per]))
#                 self.total_processes += 1
#         else:
#             self.data.append(self.pool.map(self.do_working, iter))
#             self.total_processes += 1
#         for i in range(self.total_processes):
#             adata = list(self.data[i])
#             print('{} SAVED: {}'.format(len(adata), adata))
#             self.complete(adata)

#     def do_working(self, code):
#         raise NotImplementedError("需要在子类中实现该方法")
