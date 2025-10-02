Adding new modality
===================

Structure of the repository:

.. code-block::

    src
    └── multimeditron
        ├── cli
        ├── config
        ├── dataset
        │   └── loader
        │       └── image
        ├── model
        │   ├── modalities
        │   └── projectors
        ├── train
        ├── utils
        └── verl

In order to add a new modality, we must first understand how the training pipeline process raw modalities:

.. graphviz::

   digraph G {
       rankdir=LR;

       A [label="Raw modality"];
       B [label="torch.Tensor"];
       C [label="torch.Tensor"];

       A -> B [label="modality processing"];
       B -> C [label="modality embedding"];
   }

