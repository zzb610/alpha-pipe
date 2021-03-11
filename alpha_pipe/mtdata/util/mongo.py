import os

def restart_mongo(mongo_log_dir, db_dir, docker_name=None):
    cmd = None
    if docker_name is None:
        cmd = 'mongod --fork --logpath {} --dbpath {}'.format(
            mongo_log_dir, db_dir)
    else:
        # cmd = 'docker exec {} mongod --fork --logpath {} --dbpath {}'.format(
        #     docker_name, mongo_log_dir, db_dir)
        cmd = 'docker start {}'.format(docker_name)
    status = os.system(cmd)
    return status
