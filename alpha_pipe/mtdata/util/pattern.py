def singleton(cls):
    # 单例
    instances = {}

    def _wrapper(*args, **kwargs):

        if cls not in instances:

            instances[cls] = cls(*args, **kwargs)

        return instances[cls]

    return _wrapper
