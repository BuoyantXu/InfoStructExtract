import pandas as pd

from batch.batch_steps import step_create_batches, step_upload_batches, step_download_output, step_merge_output, \
    remove_batch_files

if __name__ == "__main__":
    # Example 2: batch API processing
    df = pd.read_pickle(r"examples/example_data.pkl")

    # step 1. create batches
    # create .jsonl files in "batch/batch_input"
    step_create_batches(df['文本'])

    # step 2. upload batches
    step_upload_batches()

    # step 3. download batches
    step_download_output()

    # step 4. merge output
    step_merge_output()

    # reset the batch files
    remove_batch_files()
