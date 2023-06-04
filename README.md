[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11122310&assignment_repo_type=AssignmentRepo)
# Music genre classification
Classificació d’arxius .au situats a la carpeta gtzan, ja dividits per train, test i validació amb dades balencejades.

Hi ha una implementació principal en pytorch en la gran majoria de models, però també hi ha una implementació en keras o pytorch lighting poc desenvolupada, sent les mateixes que hi havien a l’starting point.

Ús de models RNN per fer aquesta classificació: RNN simple, GRU, LSTM, CNN + LSTM i CNN + GRU.

# Data
Característiques extretes dels arxius .au per poder entrenar el model:

[MFCC] (https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
[Spectral Centroid] (https://wikipedia.org/Wiki/Spectral_centroid)
[Chroma] (https://laborsa.ee.columbia.edu/matlab/chroma-ansyn/)
[Spectral contrast] (https://ieeexplore.ieee.org.document/1035731/)

Aquestes característiques s’extreuen amb l'arxiu GenreFeatureData.py amb la llibreria de python librosa.

L’execució del codi per poder realitzar Data augmentation és el següent:

1. DataAugmentation.py → arxiu de prova per realitzar que es pot aplicar data augmentation a un arxiu per veure si funciona o no.

2. DataAugmentation_for.py → arxiu que crea arxius nous a través del que existeix en el train però amb soroll de fons.

3. RemoveDataAugmentation → arxiu que elimina tot el data augmentation que s’ha pogut creat abans, només deixant l'arxiu original.

# Dependències
A l’arxiu de requeriments estan totes les dependències per poder executar el codi. Recomanació per fer la seva instal·lació:

	# pip install -r requirements.txt

# Requirements
La versió recomanada per l’execució dels codi és a partir de la 3.8, a part de tenir les llibreries actualitzades.
- Matplotlib
- Pytorch
- Pytorch-lightning
- Torchvision
- Keras
- Numpy 
- Wandb
- Sklearn
- Librosa

# Execució
Per tal de poder executar els diferents models implementats:
- S’ha d’estar al directori xnap-project-ed_group_06.
- Després fer un cd a LSTM-Music-Genre-Classification-master/LSTM-Music-Genre-Classification-master/.
- Realitzar Data augmentation si és necessari.
- Executar el model desitjat.

# Models
Tots els models tenen com  a configuració determinada, 400 èpoques d’execució, batch size de 35, batch normalization, optimizer ADAM, lr de 0.001, weight decay, dropout i inicialització de pesos. Menys el lstm_genre_classifier_pytorch_lightning.py i lstm_genre_classifier_keras.py, que són arxius per desenvolupar el projecte amb altres llibreries que no s’han arribat a utilitzar. 

# Accuracy
El millor accuracy obtingut ha estat del CNN + GRU, implementat a l’arxiu GRUEncoder.py, amb els següents resultats:

Loss train --> 0.3296

Loss val --> 1.1047

Train Accuracy --> 89.52 

Val Accuracy --> 70.48


## Contributors
Adrià Baldevell Comajuan (1604543@uab.cat), Adán Jiménez López (1606338@uab.cat) i Joan Paz Garcia (1598851@uab.cat)

Xarxes Neuronals i Aprenentatge Profund
Grau d'Enginyeria de Dades, 
UAB, 2023
