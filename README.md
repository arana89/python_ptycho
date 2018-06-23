# python_ptycho
NOTES to run python jobs on hoffman2:
1. For parallel scripts, install mpi4py on hoffman2 $HOME path under python version 2.7.13 https://www.hoffman2.idre.ucla.edu/software/python/#Install_Python_packages_in_your_HOME_directory
2. 'chmod +x script.py' to make executable
3. use job.q to build serial command file, or intelmpi.q to build parallel command file, '-n' parameter specifies number of workers. This must equal the number of modes being reconstructed. Also, various optional arguments are passed to ePIE here (beta_obj, beta_ap etc)


