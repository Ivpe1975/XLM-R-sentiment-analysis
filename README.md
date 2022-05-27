__   __ _     ___  ___      ______   _____            _   _                      _      ___              _           _     
\ \ / /| |    |  \/  |      | ___ \ /  ___|          | | (_)                    | |    / _ \            | |         (_)    
 \ V / | |    | .  . |______| |_/ / \ `--.  ___ _ __ | |_ _ _ __ ___   ___ _ __ | |_  / /_\ \_ __   __ _| |_   _ ___ _ ___ 
 /   \ | |    | |\/| |______|    /   `--. \/ _ \ '_ \| __| | '_ ` _ \ / _ \ '_ \| __| |  _  | '_ \ / _` | | | | / __| / __|
/ /^\ \| |____| |  | |      | |\ \  /\__/ /  __/ | | | |_| | | | | | |  __/ | | | |_  | | | | | | | (_| | | |_| \__ \ \__ \
\/   \/\_____/\_|  |_/      \_| \_| \____/ \___|_| |_|\__|_|_| |_| |_|\___|_| |_|\__| \_| |_/_| |_|\__,_|_|\__, |___/_|___/
                                                                                                            __/ |          
                                                                                                           |___/           


Instructions to reproduce the resutls:
1) Start with downloading the Amazon reviews dataset from AWS( https://registry.opendata.aws/amazon-reviews-ml). It was too big to host on this github repo. Place the json folder into the same folder as ass the scripts. We have created the /json/dev/ folder in this repo to demonstrate where the files should be.
1) You can reproduce the baseline model by running the baseline.py script. This will yield a baseline_metrics.txt file which will hold the f1 scores and the accuracies. 
1) Running any of the roberta_xx.py scripts will yield a model fine tuned on language xx and a results_xx.txt file which will hold all the f1 scores/accuracies for the fine-tuned model. The roberta_xx.job files are there if you are running it on the HPC cluster.
3) The lang2vec notebook contains all of the calculations for the mean R^2 values of the distance types. Note that the f1 scores are manually input into the notebook by defining the results vector.
4) The analysis notebook contains the pipeline for our quantatative and qualitative analysis. Before running it you will need to run roberta_de_test.py in order to get the wrong_ids, y_true and the y_pred vectors. In this case this is analysis of a sample model(fine-tuned on German predicting English)
