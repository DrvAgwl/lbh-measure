# Blank Python Project
Sample Project Template for Python Services in Udaan

### Note:
- This service architecture is mainly focussed to deploy DS/ML Model Services but can be easily modified as per other use-case needs
- This service uses github actions for CI/CD
_[*Recommended*]_ To leverage the standard DS/ML Slack Notifications Pipeline reach-out to [ML Engineering]( https://github.com/orgs/udaan-com/teams/ml-eng/members ) on `#ml-eng-products` channel
  - Add `DS_SLACK_TOKEN` secret contains slack bot token to repo secrets, by default build-script will fail without it
  - Scheduled Update will be enabled by default when this template is used, to disable it 
    - Go to `Actions > Click on Scheduled Model Update(Workflow) > Click on three dots(...) > Choose Disable Workflow`


### Best Practices [ _Strongly Recommended_ ]
- Each service should install packages only via Pipfile using `pipenv install <package-name>`
- `requirements.txt` should be generated automatically from Pipfile by running the following command
  - `pipenv lock -r | awk 'NR > 8 { print }' > requirements.txt` in the project home


### Run in Local
- `cd resources`
- `sh dev-server.sh`
