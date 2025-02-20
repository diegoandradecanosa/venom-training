In this section we show the programs and routines executed to obtain the results seen in the project
  
In order to exectute any experiment, you must activate the python enviroment created for these probes. So, you have to execute in tour shell the script `./create_python_enviroment.sh` and then activate the enviroment with `source python_enviroment/bin/activate`. 

The scripts and programs included in this folder are explained in the following lines:
* `sparsified_model_distributed.py`: The program that executes the training and save the information using the profiler. SLURM variables, `BATCH_SIZE` and `NOMBRE_TRAZA` must be defined to be executed properly.
* `sparse_distributed.sh`: Bash script that interacts with SLURM and define `BATCH_SIZE` and `NOMBRE_TRAZA` as the inputs of the user. An example of use could be `sbatch sparse_distributed.sh 32 traces_batch-size-32`
* `setup.sh`: Script that creates the python enviroment used in these analysis.
* `extract_times_from_traces.py`: Program that extract the times of the steps, GEMM kernels and spatha kernels. It generates a json with the results.
* `recipe_example.yaml': SparseML recipe with the VENOM modifier at the epoch 25.
* `graphics.py`: Python program used (out of the test enviroment) to create the report figures.

The workflow used is the following: We used sparse_distributed.sh to generate the traces and with extract_times_from_traces.py we can use external development enviroments (like Spyder or Jupyter Notebook) and the program graphics.py to see the performance of the model in different configurations.

Finally, in order to use tensorboard, you must execute the command `pip install torch-tb-profiler` in the python enviroment. Using `tensorboard --host <your-host> --logdir <trace-directory>` you can open a tensorboard profiler enviroment to see additional results used in our informs
