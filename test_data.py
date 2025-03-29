def get_test_email():
    """
    Returns:
        a dict structure with from, to, subject and body fields
    """
    return {
        "from": "Alice Smith <alice.smith@company.com>",
        "to": "Nikolaos Abatzis <nikolaos.abatzis@nouss.com>",
        "subject": "Quick question about API documentation",
        "body": """
            Hi John,

            I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

            Specifically, I'm looking at:
            - /auth/refresh
            - /auth/validate

            Thanks!
            Alice""",
    }
