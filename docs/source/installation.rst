.. role:: tmmgreen

Installation
============

Before using :tmmgreen:`TMMax`, you need to install it on your system. This section provides detailed instructions for installing :tmmgreen:`TMMax` using two common approaches: via PyPI (**recommended for simplicity**) or by cloning the repository and installing locally (**recommended for development or contributing**).

Installing via PyPI (**Recommended**)
---------------------------------

The easiest way to install :tmmgreen:`TMMax` is through PyPI, the Python Package Index. This ensures that you get the latest stable release and all necessary dependencies automatically.

1. Open a terminal or command prompt.
2. Execute the following command:

.. code-block:: bash

   pip3 install tmmax

3. After installation, verify that TMMax is installed correctly by running:

.. code-block:: python

   import tmmax
   print(tmmax.__version__)

Installing from the Source (Local Installation)
-----------------------------------------------

If you want to work on the TMMax source code, contribute, or use the latest development version, you can clone the repository and install it locally. This method allows you to directly modify the code and immediately test changes.

1. Clone the TMMax repository:

.. code-block:: bash

   git clone https://github.com/bahremsd/tmmax.git

2. Navigate into the cloned repository:

.. code-block:: bash

   cd tmmax

3. (Optional) Create and activate a Python virtual environment.  
   If you do not already have a virtual environment, create one using the following command:

.. code-block:: bash

   python3 -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate

4. Install TMMax and its dependencies in editable mode:

.. code-block:: bash

   pip install -e .

5. Verify the installation:

.. code-block:: python

   import tmmax
   print(tmmax.__version__)

Notes
-----

* Using a virtual environment is recommended to avoid conflicts with other Python packages.
* The PyPI installation is ideal for general use, while local installation is more suitable for developers or users who need the latest updates.
* Ensure that your Python version is compatible with :tmmgreen:`TMMax` (Python 3.10+ is recommended).

Once installed, you can start exploring :tmmgreen:`TMMax` functionalities.