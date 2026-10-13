"""Rendering a world state to an image, so it can be looked at rather than only measured.

Every check in this project is numeric, and numeric checks miss a whole class of problem: every agent hugging
a wall, sources clustered in one corner, a spawn distribution that is not what anyone intended. Those are
obvious in one glance and invisible in a passing suite.

**An unverified renderer is worse than no renderer**, because a wrong one is trusted and corroborates
whatever you already believed. Obstacles drawn transposed, or agent positions off by one, would make you
confident about a world that is not the world being simulated. So the correctness test asserts against the
rendered array rather than against how the picture looks, and it is the point of this module rather than an
addendum to it.

`frame` renders one state. Trajectory playback is deliberately not built: sampled trajectories do not exist
until build step 7, and a consumer for data that does not exist is written against imagined output. Playback
should be a loop over `frame` when it arrives.
"""

from src.render.frame import CHANNEL, render_state  # noqa: F401
