# 'chat-api'

'chat-api' is new micro service
Behold My Awesome Project!

# Setting up the project
Please delete this part after setting up whole project 

# Setting the init version
1. Delete Readme about project setup
2. Initialize git and add origin:
```bash
git init
git add .
git commit -m 'Init API'
git remote add origin git@bitbucket.org:chat_api/chat-api.git
git push -u origin master
```

# Setting environment files

1. Add a role `Secret Manager Secret Accessor` to the App Engine default service account

2. Create a `chat_api_django_settings` entry in the google secret manager in the beta 
and prod GCP projects containing the variables: 

```
CHAT_API_DB_USER=
CHAT_API_DB_NAME=
CHAT_API_DB_PASSWORD=
CHAT_API_DB_PORT=
CHAT_API_DB_HOST=
CHAT_API_DB_GCP_HOST=

REACT_APP_URL=

SENDGRID_API_KEY=
SENDGRID_FROM_EMAIL=
SENDGRID_RESET_PASSWORD_TEMPLATE_ID=
SENDGRID_INVITATION_TEMPLATE_ID=
```

The env file will be populated automatically during the deployment.

# Local usage
1. Setup git to ignore formatting revs `git config blame.ignoreRevsFile .git-blame-ignore-revs`
2. We use Black linter. Install pre-commit with `pre-commit install`
3. Run all pre-commit hooks on a repository with `pre-commit run --all-files`
4. (optional) Provide authentication credentials to your application. Create `client-secret.json` file in the main
   folder with the certificate. Then create an environment variable ```GOOGLE_APPLICATION_CREDENTIALS``` and set an
   absolute path to your `client-secret.json` file as a value.
5. (optional) If you want to use custom firebase cert you need to create
   `firebase-cert.json` file in the main folder with the certificate
6. Set up local environment `cp .env.local .env`
3. Setup Local Database:
    - `docker-compose up db`
    
# Deployment
Deployments are triggered when you create a new release-* tag (for example release-1.0.0).
Beta deployment is completely automatic and to deploy to prod you need to click a button in the BitBucket pipeline.

# Connecting to the production db
You can use for that sql proxy, instruction under this link:
https://cloud.google.com/python/django/appengine

#Logging
Loggers are logging to stackdriver logging and stackdriver error reporting when project is deployed to beta and prod.
Error reporting is getting logs on level ERROR and higher.
Locally loggers are logging only to local python console.

#Profiling

With DEBUG=True set in env you can add `?prof` to your HTTP queries to receive
a profiler output that will help you determine what part of the code takes the
most time to execute. You can also sort the results by different properties, read more at
https://docs.python.org/2/library/profile.html#pstats.Stats.sort_stats