# models/

Place your pretrained nnU-Net v2 model checkpoint(s) and plans/config files
here once available (for example the `nnUNetTrainer__nnUNetPlans__3d_fullres`
folder that nnU-Net v2 produces after training).

This folder is intentionally empty in the frontend-only build. Nothing in
the application requires files here yet — `backend/inference.py` is where
you will load a checkpoint from this folder once real inference is wired up.
