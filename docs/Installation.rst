Installation
============
Hydrostats is freely available on the Python Package index repository (PyPI). It can be installed
with the following command using either virtualenv or Anaconda.

.. code-block:: shell

   pip install hydrostats

When installing hydrostats on Mac OSX operating system, you may get the following error when trying
to run python scripts using hydrostats in an IDE such as PyCharm.

.. code-block:: shell

   **RuntimeError**: Python is not installed as a framework. The Mac OS X backend will not be able
   to function correctly if Python is not installed as a framework. See the Python documentation for
   more information on installing Python as a framework on Mac OS X. Please either reinstall Python
   as a framework, or try one of the other backends.

If this happens, you will need to manually change the backend for use in your IDE. This can be done
by running the following commands in the terminal:

.. code-block:: shell

   touch ~/.matplotlib/matplotlibrc  # Create a file called matplotlibrc

Add the following snippet of text to the file you created:

.. code-block:: shell

   backend: TkAgg  # or whatever backend you would like to use

Then save and close the document.

For more information about backends, read here_

.. _here: https://matplotlib.org/faq/usage_faq.html#what-is-a-backend
