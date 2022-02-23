.. _labeled_faces_in_the_wild_dataset:

The Labeled Faces in the Wild face recognition dataset
------------------------------------------------------

This dataset is a collection of JPEG pictures of famous people collected
over the internet, all details are available on the official website:

    http://vis-www.cs.umass.edu/lfw/

Each picture is centered on a single face. The typical task is called
Face Verification: given a pair of two pictures, a binary classifier
must predict whether the two images are from the same person.

An alternative task, Face Recognition or Face Identification is:
given the picture of the face of an unknown person, identify the name
of the person by referring to a gallery of previously seen pictures of
identified persons.

Both Face Verification and Face Recognition are tasks that are typically
performed on the output of a model trained to perform Face Detection. The
most popular model for Face Detection is called Viola-Jones and is
implemented in the OpenCV library. The LFW faces were extracted by this
face detector from various online websites.

**Data Set Characteristics:**

    =================   =======================
    Classes                                5749
    Samples total                         13233
    Dimensionality                         5828
    Features            real, between 0 and 255
    =================   =======================

Usage
~~~~~

``scikit-learn`` provides two loaders that will automatically download,
cache, parse the metadata files, decode the jpeg and convert the
interesting slices into memmapped numpy arrays. This dataset size is more
than 200 MB. The first load typically takes more than a couple of minutes
to fully decode the relevant part of the JPEG files into numpy arrays. If
the dataset has  been loaded once, the following times the loading times
less than 200ms by using a memmapped version memoized on the disk in the
``~/scikit_learn_data/lfw_home/`` folder using ``joblib``.

The first loader is used for the Face Identification task: a multi-class
classification task (hence supervised learning)::

  >>> from sklearn.datasets import fetch_lfw_people
  >>> lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

  >>> for name in lfw_people.target_names:
  ...     print(name)
  ...
  Ariel Sharon
  Colin Powell
  Donald Rumsfeld
  George W Bush
  Gerhard Schroeder
  Hugo Chavez
  Tony Blair

The default slice is a rectangular shape around the face, removing
most of the background::

  >>> lfw_people.data.dtype
  dtype('float32')

  >>> lfw_people.data.shape
  (1288, 1850)

  >>> lfw_people.images.shape
  (1288, 50, 37)

Each of the ``1140`` faces is assigned to a single person id in the ``target``
array::

  >>> lfw_people.target.shape
  (1288,)

  >>> list(lfw_people.target[:10])
  [5, 6, 3, 1, 0, 1, 3, 4, 3, 0]

The second loader is typically used for the face verification task: each sample
is a pair of two picture belonging or not to the same person::

  >>> from sklearn.datasets import fetch_lfw_pairs
  >>> lfw_pairs_train = fetch_lfw_pairs(subset='train')

  >>> list(lfw_pairs_train.target_names)
  ['Different persons', 'Same person']

  >>> lfw_pairs_train.pairs.shape
  (2200, 2, 62, 47)

  >>> lfw_pairs_train.data.shape
  (2200, 5828)

  >>> lfw_pairs_train.target.shape
  (2200,)

Both for the :func:`sklearn.datasets.fetch_lfw_people` and
:func:`sklearn.datasets.fetch_lfw_pairs` function it is
possible to get an additional dimension with the RGB color channels by
passing ``color=True``, in that case the shape will be
``(2200, 2, 62, 47, 3)``.

The :func:`sklearn.datasets.fetch_lfw_pairs` datasets is subdivided into
3 subsets: the development ``train`` set, the development ``test`` set and
an evaluation ``10_folds`` set meant to compute performance metrics using a
10-folds cross validation scheme.

.. topic:: References:

 * `Labeled Faces in the Wild: A Database for Studying Face Recognition
   in Unconstrained Environments.
   <http://vis-www.cs.umass.edu/lfw/lfw.pdf>`_
   Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller.
   University of Massachusetts, Amherst, Technical Report 07-49, October, 2007.


Examples
~~~~~~~~

:ref:`sphx_glr_auto_examples_applications_plot_face_recognition.py`
