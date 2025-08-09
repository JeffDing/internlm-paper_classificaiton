from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = 'YOUR_ACCESS_TOKEN'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

owner_name = 'owner_name'
dataset_name = 'dataset_name'

api.upload_file(
    path_or_fileobj='arxiv_dataset/arxiv-metadata-oai-snapshot.json',
    path_in_repo='arxiv-metadata-oai-snapshot.json',
    repo_id=f"{owner_name}/{dataset_name}",
    repo_type = 'dataset',
    commit_message='upload dataset file to repo',
)
