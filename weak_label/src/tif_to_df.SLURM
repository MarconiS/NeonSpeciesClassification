#!/bin/bash
#SBATCH --job-name=tif_to_df    # Job name
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sergio.marconi@weecology.org   # Where to send mail
#SBATCH --account=ewhite
#SBATCH --qos=ewhite-b

#SBATCH --cpus-per-task=1            # Number of cores per MPI rank
#SBATCH --mem=6gb

#SBATCH --time=12:00:00       #Time limit hrs:min:sec
#SBATCH --output=tif_to_df.out   # Standard output and error log
#SBATCH --error=tif_to_df.err


pwd; hostname; date
ml conda
source activate quetzal3

python3 ./weak_label/src/bash_tif_to_df.py $1 
echo "done"

date
