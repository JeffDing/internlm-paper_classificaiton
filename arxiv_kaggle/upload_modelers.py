from openmind_hub import upload_file

upload_file(
    path_or_fileobj="arxiv_dataset/arxiv-metadata-oai-snapshot.json",
    path_in_repo="arxiv-metadata-oai-snapshot.json",
    repo_id="username/repo",
    token="token",
    revision="main",
    commit_message="upload file",
    commit_description=None,
)
