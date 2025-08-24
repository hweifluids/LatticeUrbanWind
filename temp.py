
from dask.distributed import Client, LocalCluster
c = Client(LocalCluster())
print(c)
c.close()
PY
