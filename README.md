# CS5470
Welcom to CS5470! This repo contains the assignments that you are going to complete in this course.

## Working on Perlmutter
[Getting Started at NERSC](https://docs.nersc.gov/getting-started/) gives a very helpful overview of Perlmutter. Please make sure to read the documentation.

When you first connect to Perlmutter, you will be on the login node! This is the place where you install environments, libraries, ane compile your workloads. This is NOT the place to execute jobs, such as training, fine-tuning, or serving workloads. You should only run computation on compute nodes through the interactive queue or (better) batch [script](https://my.nersc.gov/script_generator.php).

If you have any questions about the account process or system, please feel free to contact the NERSC help desk via their [ticket system](https://nersc.servicenowservices.com/sp), or post on Ed.

## Common workflow
```bash
# On login node
module load conda
conda env create ...
pip install ...
make ...

# Allocate GPU nodes
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus <gpu_num> --account <projectID>

# On GPU nodes
python ...
vllm ...
torchrun ...
```
