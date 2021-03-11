from time import time
from .pattern import singleton


@singleton
class Cache(object):
    """
    简单的缓存系统
    把这个变量保存在内存里, 同时给它一个过期时间, 过期则失效.
    """

    def __init__(self):
        # 存储数据
        self.mem = {}
        # 到期时间
        self.time = {}

    def set(self, key, data, age=-1):
        """设置到期时间

        Parameters
        ----------
        key :
            数据键
        data : [type]
            数据
        age : int, optional
            到期时间, by default -1, -1表示永久保存
        """
        self.mem[key] = data
        if age == -1:
            self.time[key] = -1
        else:
            self.time[key] = time() + age
        return True

    def get(self, key):
        """获取缓存数据

        Parameters
        ----------
        key : 
            数据键

        Returns
        -------
            缓存数据
        """
        if key in self.mem.keys():
            if self.time[key] == -1 or self.time[key] > time():
                return self.mem[key]
            else:  # 过期, 删除数据
                self.delete(key)
        return None

    def delete(self, key):
        """根据键值删除缓存

        Parameters
        ----------
        key :
            键值
        """
        del self.mem[key]
        del self.time[key]
        return True

    def clear(self):
        """清空所有缓存
        """
        self.mem.clear()
        self.time.clear()
