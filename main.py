import pandas as pd

from utils import step_create_batches, step_upload_batches, step_download_output, step_merge_output, remove_batch_files

if __name__ == "__main__":
    df = pd.read_pickle(r"example_data.pkl")

    # 1. create batches
    # create in .jsonl files in "batch/batch_input"
    step_create_batches(df['文本'])

    # 2. upload batches
    step_upload_batches()

    # 3. download batches
    step_download_output()

    # 4. merge output
    step_merge_output()

    # reset the batch files
    remove_batch_files()
