.. MultiMeditron documentation master file, created by
   sphinx-quickstart on Wed Oct  1 14:49:37 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. Placeholder for a cooler banner
.. image:: _static/light_logo.png
    :alt: LiGHT logo
    :align: center
    :width: 200px

MultiMeditron documentation
###########################

🎉 Latest Updates
=================

2025/09:
    - First version of the MultiMeditron pipeline!


✨ Overview
===========

MultiMeditron is a scalable and modular pipeline to train multimodal models.

Features:

- **Modular modality**: Designed to be easily expanded to any types of modality
- **Scalable**: Scalable to multinode training and efficient GPU memory usage using Deepspeed
- **Configurable**: Trainings can be configured using a single YAML configuration file
- **Easy to install**: We provide Docker images for easier reproducibility


📚 Documentation
================

.. toctree::
    :glob:
    :maxdepth: 1

    User Guide <guides/guide>
    Reference <ref/modules>


