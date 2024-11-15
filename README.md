BEFORE STARTING CHECK PATHS OF ALL FILES, THEY'RE SET TO MINE AND ALSO SOME MIGHT BE SET TO TEST FILES NOT THE BIG ONES
1. Follow the instructions from this tutorial https://github.com/curlsloth/NYU-HPC-4-newbies including 5.1 so you have a script that start you're singularity
2. Download the TSCC data and then open data/data_clean_up_and_split.ipbyn locally
  a. The first chunk takes in a folder of tsv files, gets rid of the nWords column (necessary for some older code to work) and returns a folder of tsv files
  b. The second chunk takes in a folder of tsv files and returns 3 folders with the data split into train, test and split
3. Use git clone ssh (not http) when on greene to get access to the repo 
4. Use the scp command to tranfer the data files into the data folder from the repo. It's always scp folder_from folder_to
  scp (-r for a folder) filename (or . For when transferring a whole folder and you're in it)  netid@greene.hpc.nyu.edu:/scratch/netid/preferred folder   
5. Make sure your paths work with the whole setup 
6. Edit the run_proj.bash file so its you're named overlay from 5.1, mine is called my_sing.ext3
7. Check to make sure everything is working by using an interactive node:
  a. srun --nodes=1 --tasks-per-node=1 --cpus-per-task=16 --mem=32GB --time=1:30:00 --gres=gpu:1 --pty /bin/bash!
  b. /scratch/<netid>/your_working_folder/run_proj.bash  <- this starts up your singularity and conda env
  c. From the comp_ling_proj directory run the command: python src/scripts/run_llama.py --data_dir "/scratch/ah5192/comp_ling/comp_ling_proj/data/tscc_split/tiny" --output_dir "/scratch/ah5192/comp_ling/slurm_output"
  d. You might also want to go an uncomment the print statements in run_llama.py, but this should output a jsonl file
8. Change the file paths in sbatch_proj.s, first call is to you .bash file, second is to run_llama - might want to try it on tiny first to see if things work then try on train or a section of train
   a. I'm not sure if we get output if it times out - either see if you can change the code in run_llama or make a new folder that only has a few examples from train and submit that
   b. I'm also not sure if the sbatch script is working correctly
9. scp the output from greene to your local machine and upload it to github under data

Note: my greene has been weird recently but you might need to do two scp call when the singularity is running: the first is from your local machine to greene like above and the second is from the singularity to greene like this
scp (-r for a folder) greene-dtn:/scratch/netid/location_on_greene /scratch/netid/location you want it on the singularity
