.. role:: bash(code)
   :language: bash


Known issues
============

This section describes the list of known issues and possible fix

.. _docker-permission:

Wrong permissions with Docker images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are running the installation with Docker image on HPC. You will most likely mount volumes with specific permissions. When running with Docker image, you might see:

.. code-block::

   I have no name!:~$

or

.. code-block:: 

   root:~$

and get a permission denied.

If you get such problem try the following fix:

.. code-block:: bash

    groupadd -g $GID  $GROUPNAME
    useradd -u $UID -g $GID $USERNAME
    su $USERNAME

You can get your `$UID` or `$GID` by connecting on the login node and running:

.. code-block:: bash

    id -u
    id -g

