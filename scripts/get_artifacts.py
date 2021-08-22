import argparse
import os
import requests
import json
from datetime import datetime


class github_session:
    def __init__(self, token, repository='rurabori/diploma_research', remote_url='https://api.github.com') -> None:
        self.token = token
        self.repository = repository
        self.remote_url = remote_url

    def get_all_artifacts(self):
        response = requests.get(f'{self.remote_url}/repos/{self.repository}/actions/artifacts', headers={
            'Authorization': f'token {self.token}', 'Accept': 'application/vnd.github.v3+json'})

        return json.loads(response.text)

    def download_artifact(self, artifact, output_path):
        response = requests.get(artifact['archive_download_url'], headers={
            'Authorization': f'token {self.token}'})

        with open(output_path, 'wb') as fout:
            fout.write(response.content)


def parse_github_datetime(as_string: str):
    return datetime.strptime(as_string, '%Y-%m-%dT%H:%M:%SZ')


def main(args):
    session = github_session(args.git_token)

    all_artifacts = session.get_all_artifacts()

    with_name = [*filter(lambda artifact: artifact['name']
                         == args.artifact_name, all_artifacts['artifacts'])]

    if len(with_name) == 0:
        print(
            f'there are no artifacts with matching name({args.artifact_name}), aborting')
        exit(1)

    in_order = sorted(with_name, key=lambda artifact: parse_github_datetime(
        artifact['updated_at']), reverse=True)

    output_path = args.output_path if args.output_path else f'{args.artifact_name}.zip'
    session.download_artifact(in_order[0], output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--git-token',
                        default=os.environ.get("GITHUB_TOKEN"))
    parser.add_argument('-n', '--artifact-name', default='debian-10')
    parser.add_argument('-o', '--output-path')

    main(parser.parse_args())
