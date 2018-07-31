Toplogy
=======

The code developed here is, perhaps, best described through the lens
of the project for which the code was originally developed. If there
are any questions, please contact the original author, Branden Kappes
<bkappes@mines.ed>.

This code was originally developed to identify the lowest cost path
through an atom field (crystal structure) for a lithium ion battery
anode material. The concept is simple: a guest atom (lithium)
introduced into a fully dense host material will dilate the host. As a
first pass, then, the interaction between the guest and atoms in the
host will be purely repulsive. Therefore, the path that maximizes the
distance in the crystal structure will minimize the energy. (This
path, then, could be used in more detailed calculations, e.g. nudged
elastic band (NEB) in density functional theory (DFT).)

Description of the code logic
-----------------------------

(The codes, found in *src/topology*, are not in a form to be readily
used, but can be readily modified for a specific application. Please
contact the author to work to develop a more robust code base.)

A regular grid (voxels) is superimposed over a field variable, e.g. a
field of atom positions. A property is calculated from each voxel to a
point in the field. (In this example, the distance from that voxel to
the nearest atom position.) This is handled through the python wrapper
*distance.py*.
