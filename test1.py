import socket

from mpi4py import MPI

hostname=socket.gethostname()
comm=MPI.COMM_WORLD

print(comm)

rank=comm.Get_rank()
p_name=MPI.Get_processor_name()
all_hostnames=comm.allgather(p_name)
print(rank)
print("hostname:",all_hostnames)