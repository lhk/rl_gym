So far I'm using pygame to render my homebrew environments.
But rotations don't support pixel wise collisions in an elegant way.
Also, I think that the conversions between pygame surfaces and numpy arrays are not efficient.

So this directory contains a very small toolkit to draw images onto numpy arrays.
The key feature is that the images can be rotated and the tool checks for pixel-precise collision.

I'll have to benchmark this and see how it compares against pygame.