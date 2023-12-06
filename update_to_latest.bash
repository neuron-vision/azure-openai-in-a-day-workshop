#git stash
git pull origin main
python3.11 -m pip install virtualenv
python3.11 -m virtualenv /workspaces/azure-openai-in-a-day-workshop/venv
source /workspaces/azure-openai-in-a-day-workshop/venv/bin/activate
/workspaces/azure-openai-in-a-day-workshop/venv/bin/pip install -r requirements.txt

