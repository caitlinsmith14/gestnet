## GestNet

On this page, you'll find the supplementary materials for the following paper:

Smith, Caitlin, Charlie O'Hara, Eric Rosen, Paul Smolensky (2021) [Emergent Gestural Scores in a Recurrent Neural Network Model of Vowel Harmony](https://pages.jh.edu/~csmit372/pdf/smithohara_scil2021_paper.pdf). To appear in *Proceedings of the Society for Computation in Linguistics: Volume 4*.

This paper introduces GestNet, a sequence-to-sequence, encoder-decoder recurrent neural network architecture in which a string of input symbols is translated into sequences of vocal tract articulator movements. In its first application, we task our models with learning stepwise (chain-shifting) height harmony.

### Code Walkthrough

The code for GestNet can be found in `gestnet.py` above. Here, we will walk you through how to use this code to create, train, and evaluate your own GestNet model(s). We have not released any pretrained models at this time.

**Check python version and install dependencies.** The code for GestNet is written for use with python 3.8. It is likely compatible with some older versions of python 3, but we make no guarantees. The code uses several packages that are not included in the python standard library and must be installed by the user.

GestNet is written using `pytorch` (conda/pip install) version 1.6.0. It is likely compatible with some older and newer versions of `pytorch`, but again, we make no guarantees.

GestNet's `Dataset` class uses a slightly older version of `torchtext` (pip install): version 0.7.0. Many of the features of this package have been deprecated in more recent versions of `torchtext`, so be sure to specify version 0.7.0 when installing. We will likely update the `Dataset` class in the future to be compatible with more recent versions of `torchtext`.

In addition, GestNet utilizes the packages `matplotlib` (pip install) and `tqdm` (pip install).

**Import GestNet.** To get started with GestNet, first import all of the classes in `gestnet`:

`>>> from gestnet import *`

**Create a Dataset object.** When you import from `gestnet`, a premade `Dataset` object called `data_stepwise` is created for you from the text file `trainingdata_stepwise.txt`. If you'd like to manually create this dataset, use the following command:

`data_stepwise = Dataset('path/to/dataset')`

**Create a new Seq2Seq model.** All GestNet models are objects from the `Seq2Seq` class. To create a new model, you must provide as input the `Dataset` object containing the training data, as well as several parameter values: size of the segment embedding vector, size of the hidden layer, size of the attention vector, type of optimizer, and learning rate.

```
>>> model = Seq2Seq(training_data=data_stepwise,
                    seg_embed_size,  # size of segment embedding
                    hidden_size,  # size of hidden layer
                    enc_dec_attn_size,  # size of encoder-decoder attention vector
                    optimizer,  # optimizer ('adam' or 'sgd')
                    learning_rate)  # learning rate
```
To use the default parameter values, just provide a `Dataset` object containing the training data.

`>>> model = Seq2Seq(training_data=data_stepwise)`

**Load a trained model.** To instead load a previously trained and saved model, just provide the path to the saved .pt file to the `Seq2Seq` object:

`>>> model = Seq2Seq(load='path/to/saved/model/')`

**Train a model.** Train a model for however many epochs you like with the `train_model()` method. Using all of GestNet's default settings, the model will learn the height harmony pattern in `trainingdata_stepwise.txt` in about 200 epochs.

```
>>> model.train_model(training_data=data_stepwise, n_epochs=200)
100%|██████████| 200/200 [03:47<00:00,  1.14s/it]
```

**Evaluate model performance.** There are several methods for evaluating a model's performance, both throughout and after training. To see how a model performs in predicting articulatory trajectories for all words in the dataset, use the `evaluate_model()` method. This returns the model's current average loss per word in a `Dataset` object.

```
>>> model.evaluate_model(training_data=data_stepwise)
Average loss per word this epoch:
2.173375489190221
```

To visualize how the model's average loss per word decreases with training, use the `plot_loss()` method. 

`>>> model.plot_loss()`

<img src="https://pages.jh.edu/~csmit372/pic/plotloss.png" height=300>

To see how a model performs in predicting articulatory trajectories for a particular word, use the `evaluate_word()` method. This method provides target and predicted tract variable plots, plus encoder/decoder attention maps, for an underlying form in the data.

```
>>> model.evaluate_word(training_data=data_stepwise, 'eb-i')
Target output:
tensor([[ 5., 10.],
        [ 5.,  7.],
        [ 5.,  4.],
        [ 5.,  4.],
        [ 5.,  4.],
        [-2.,  4.],
        [-2.,  4.],
        [ 5.,  4.],
        [ 5.,  4.],
        [ 5., 10.]], dtype=torch.float64)
Predicted output:
tensor([[ 5.0005,  9.4516],
        [ 4.8037,  7.2522],
        [ 4.8611,  4.0180],
        [ 4.8322,  4.0026],
        [ 4.3294,  4.2279],
        [-1.6042,  3.2091],
        [-1.9804,  4.1400],
        [ 4.5110,  4.0593],
        [ 4.7825,  4.1043],
        [ 4.8601,  9.0935]])
Encoder Decoder Attention:
tensor([[2.3660, 0.3572, 0.0395],
        [1.1399, 0.7926, 1.9834],
        [1.2634, 0.3080, 2.2839],
        [1.5941, 0.2914, 1.6460],
        [1.8144, 0.2367, 1.2702],
        [0.5215, 0.4881, 3.5540],
        [0.2990, 2.0381, 2.2502],
        [0.0227, 1.0392, 3.9235],
        [0.0371, 0.2342, 4.8014],
        [1.5981, 0.3206, 1.6052]])
```

<img src="https://pages.jh.edu/~csmit372/pic/ebi_trajectories.png" height=300> <img src="https://pages.jh.edu/~csmit372/pic/ebi_attn.png" height=300>

**Save model.** To save a model, use the `save()` method. This method will save a model to a `saved_models` directory, and will create one if it doesn't exist already. The filename includes a sequentially assigned model number and the number of epochs the model was trained.

```
>>> model.save()
Model saved as gestnet_1_200.pt in directory saved_models.
```
