# Scene2 v2 engineering notes

v2 is a parametrised clearance probe: central slot, 0.07 m outer keepout, release 50, vertical-first withdraw, closed-gripper close screen. It is not a success-rate claim.

v1 4/8 vs v2 2/8. v2 fixed v1 layout 0 red, but release 50 still hits the tray wall by 0.50-0.59 mm on four cases; the 0.07 m keepout is insufficient at yaw 68.6 (0.535 mm); the fixed -x vertical retreat fails IK on layout 1 red. v2 loses three v1 successes.

Blind parameter shifts trade failure modes. The next v3 should search per scene over release open (40/50) and withdraw direction (vertical + both v1 sideways), and use a measured gripper-envelope keepout; then re-pilot. Do not use v2 for formal collection.

R3 local; reproduce-pending from a clean clone.
