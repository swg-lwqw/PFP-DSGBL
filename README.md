# PFP-DSGBL

PFP-DSGBL is a dual-branch method for protein function prediction. It transforms
protein function prediction into a multi-label classification task over Gene
Ontology terms by combining two independently trained branches: an ESM-2 sequence
branch (ESM2-MLP) and a relation-aware PPI graph branch (RGCN-GO) that aggregates
multi-relational interaction neighbours and is regularised by GO ontology
structure. The two branches are fused at the decision level with weights fitted
on the validation set.

# Training model
To train the models and reproduce our results:

* The dataset is distributed in the DeepGO-SE data record and can be obtained via
the following link: [10.5281/zenodo.10369249](https://doi.org/10.5281/zenodo.10369249)
- The record provides two archives; the one used here is `training-data.tar.gz`
(7.1 GB), which contains the training, validation and test splits together with
the precomputed ESM-2 features.
- The training data includes both the UniProtKB/Swiss-Prot dataset and the
neXtProt evaluation dataset.
- go.obo - Gene Ontology file
- go.norm - normal forms of GO axioms, used by the GO geometry loss
- mf, bp and cc subfolders include:
- train_data.pkl - training proteins
- valid_data.pkl - validation proteins
- test_data.pkl - testing proteins
- nextprot_data.pkl - neXtProt dataset proteins
- terms.pkl - list of GO terms for each subontology
- ppi_test.bin, ppi_nextprot.bin - PPI graphs saved with the DGL library, holding
the train/valid/test nodes of the corresponding split in a single graph

* Protein representations are ESM-2 mean representations of layer 36
(esm2_t36_3B_UR50D, truncated at 1022 residues, 2560 dimensions), stored in the
`esm2` column of the data pickles.

* Three scripts are involved. `train_mlp.py` trains the sequence branch,
`train_geo.py` trains the graph branch, and `fusion.py` fits the decision-level
fusion weights on top of the two frozen branches.

* Train the sequence branch of MFO prediction with ESM-2 embeddings by entering
the following command in the console (for details on the remaining parameters of
the command, please refer to the code): \
`python train_mlp.py -ont mf -m mlp_esm -td test`

* Train the graph branch of MFO prediction by entering the following command in
the console: \
`python train_geo.py -ont mf -m rgcn_geo -td test`

* Fit the fusion weights of MFO prediction on the validation set, using the two
branches trained above, by entering the following command in the console: \
`python fusion.py -ont mf -mg rgcn_geo -mm mlp_esm -fm fusion_geo_mlp -td test`

* Each script saves its outputs in the corresponding subfolder of the data
directory:
- mlp_esm.th, rgcn_geo.th - trained branch parameters
- test_predictions_mlp_esm.pkl, test_predictions_rgcn_geo.pkl - branch
predictions on the test set, after GO ancestor propagation
- test_predictions_fusion_geo_mlp.pkl - fused predictions on the test set

* Repeat the three commands with `-ont bp` and `-ont cc` to obtain the BP and CC
models. The fusion weights are fitted separately for each subontology.

# Use the trained model for predicting protein function
* To skip training and only run inference with the saved parameters, add `-ld` to
`train_mlp.py` and `train_geo.py`, then run `fusion.py` as above:
- `python train_mlp.py -ont mf -m mlp_esm -td test -ld`
- `python train_geo.py -ont mf -m rgcn_geo -td test -ld`
- `python fusion.py -ont mf -mg rgcn_geo -mm mlp_esm -fm fusion_geo_mlp -td test`
* To predict on the neXtProt evaluation set instead of the test split, replace
`-td test` with `-td nextprot` in all three commands.

# Evaluating the predictions
The training scripts generate predictions for the test data that are used to
compute evaluation metrics.

* To evaluate the model's prediction performance for MF, please run the
evaluate.py script and provide the model name and test data name. For detailed
information on the specific required parameters of the command, please refer to
the code. Enter the following command in the console: \
`python evaluate.py -m fusion_geo_mlp -td test -ont mf`

* The same script evaluates a single branch when the model name is set
accordingly, which is how the ablation results are produced: \
`python evaluate.py -m mlp_esm -td test -ont mf` \
`python evaluate.py -m rgcn_geo -td test -ont mf`

* The reported metrics are Fmax, Smin, AUC and AUPR, computed by scanning 101
decision thresholds over the propagated predictions.
