"""Hand-written eval set: questions about the real photo-proofing-app
repo, each paired with the file we know should be retrieved to answer it
correctly. Used to measure retrieval hit-rate -- not just "does the
system run" but "does it actually find the right code."
"""

from dataclasses import dataclass


@dataclass
class QAPair:
    question: str
    expected_file: str  # substring match against a result's file_path


QA_PAIRS: list[QAPair] = [
    QAPair("How does gallery deletion work?", "PhotographerUpload.tsx"),
    QAPair("How is a photo's selection status (selected/rejected/pending) represented in the type system?", "types/photo.ts"),
    QAPair("How does the app persist gallery data across page reloads?", "galleryStore.tsx"),
    QAPair("How is a shareable slug generated for a new gallery?", "galleryStore.tsx"),
    QAPair("How is the photographer dashboard protected from casual visitors?", "PasscodeGate.tsx"),
    QAPair("How does the app know when a client has submitted their final photo selections?", "photo.ts"),
    QAPair("How are a client's selected photos downloaded as a zip file?", "downloadZip.ts"),
    QAPair("How does the photo lightbox handle keyboard navigation (arrow keys, escape)?", "Lightbox.tsx"),
    QAPair("How are uploaded photo files shown as thumbnail previews before submitting?", "FileDropzone.tsx"),
    QAPair("How does drag-and-drop photo upload work?", "FileDropzone.tsx"),
    QAPair("How is the delete-gallery confirmation dialog implemented?", "ConfirmDialog.tsx"),
    QAPair("How does the app avoid memory leaks from object URLs created for photo previews?", "FileDropzone.tsx"),
    QAPair("How are the client gallery and photographer results routes defined?", "App.tsx"),
    QAPair("How does the app calculate how many photos are selected, rejected, and pending?", "photo.ts"),
    QAPair("What environment variable sets the dashboard passcode?", "PasscodeGate.tsx"),
    QAPair("How does the results page show progress while generating a zip download?", "PhotographerResults.tsx"),
    QAPair("How is a new gallery created from the upload form?", "PhotographerUpload.tsx"),
    QAPair("How does the copy-link button work?", "CopyLinkButton.tsx"),
]