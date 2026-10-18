from managed_deepagents import define_identity, auth

identity = define_identity(auth=auth.langsmith_api_key())
