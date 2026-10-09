# Round16R Clean Pseudo-Target Repair

Repair of Round16 TRAIN construction only; not a new method.

- source: `68d7ada0d968414b07fbd12ec1c9504aaedc8b8f`
- canonical pseudo task: `history[:-1] -> history[-1]`
- preference user retained only if pseudo target is inside canonical frozen prefix Top100
- positive and hard negative are always indexed from the same frozen Full-CoLift Top100 row
- no off-row positive context/extrapolation path
- A2/A3 gain uses direct correction margin `delta_pos - delta_neg`
- Round16 frozen BASE reference reused only after hash verification
- Test is not used for implementation, checkpoint, variant, or hyperparameter selection. User explicitly authorized one exploratory Baby Test only after Validation selection is committed and locked.
- Sports / Electronics closed.
