Optional demo / evaluation files (not required for the upload chatbot).

These are public teaching images with fictional notes, plus saved batch-run outputs.
For your own cases, use the upload UI instead:

    python main.py

or:

    python main.py --serve

Your uploads are processed under uploads/processed/ and are not mixed with this folder.

To re-run the five built-in teaching cases:

    python main.py --batch

To download or refresh the teaching images and notes:

    python main.py --batch --skip_dataset_build   (use existing files)
    python dataset_builder.py                     (force refresh)
