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

       A [label="Block 1"];
       B [label="Block 2"];
       C [label="Block 3"];

       A -> B [label="First transition"];
       B -> C [label="Second transition"];
   }

