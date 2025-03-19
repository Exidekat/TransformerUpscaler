import os


def get_latest_checkpoint(checkpoint_dir):
    """
    Search the checkpoint directory for files ending in .pth and return
    the checkpoint with the highest epoch number (assumes filenames like model_epoch_{n}.pth).
    """
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_dir}")

    def extract_epoch(filename):
        try:
            epoch_str = filename.split('_')[-1].split('.')[0]
            return int(epoch_str)
        except Exception:
            return -1

    checkpoint_files = sorted(checkpoint_files, key=extract_epoch)
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
    return latest_checkpoint, extract_epoch(latest_checkpoint)


resolutions = {
    '720': (720, 1280),
    '1080': (1080, 1920),
    '1440': (1440, 2560),
    '2k': (1440, 2560),
    '2160': (2160, 3840),
    '4k': (2160, 3840)
}
