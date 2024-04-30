Programmed in C++.

------------------------To run the program------------------------

1. log into the hpc

2. store the file in the wanted directory

3. type 

srun -p compute --mem 100G --pty /bin/bash' into the command line (easiest to do it here)

4. cd into newly-created directory

5. to compile the code into an executable, type: 

mpic++ MM_1D.cpp -o [executableName]

6. ls and make sure the executable has been created

7. to run the executable, type: 

mpirun -n [processorCount] ./[executableName] [n]

------------------------------------------------------------------

now it should run!