import os

class DistributedContext:
    def __init__(self):
        self.rank = int(os.environ.get("RANK",0))
        self.local_rank =int(os.environ.get("LOCAL_RANK",0))
        self.world_size = int(os.environ.get("WORLD_SIZE",1))
    
    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1
    
    @property
    def is_master(self) -> bool:
        return self.rank == 0